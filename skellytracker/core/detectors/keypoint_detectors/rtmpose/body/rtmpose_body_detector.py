"""RTMPose body keypoint detector (23 keypoints: COCO body + feet).

Single-stage body-only detector. Accepts an optional BoundingBox person crop;
the full image is used if none is provided. Upstream person detection (e.g.
YOLOX) is recommended for accuracy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.config.detector_configs import KeypointDetectorConfig
from skellytracker.core.data_primitives import Keypoints
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import (
    KEYPOINT_DETECTOR_REGISTRY,
    KeypointDetector,
)
from skellytracker.core.detectors.keypoint_detectors._schema_loader import load_connections, load_point_names
from skellytracker.core.detectors.keypoint_detectors.rtmpose.rtmpose_preprocessing import (
    rtmpose_letterbox_postprocess,
    rtmpose_letterbox_preprocess,
)
from skellytracker.core.detectors.metadata import RTMPoseMetadata
from skellytracker.core.sessions.model_registry import ModelSource
from skellytracker.core.sessions.onnx_session import OnnxModelSpec, OnnxSession
from skellytracker.core.sessions.session import Session

logger = logging.getLogger(__name__)

_YAML = Path(__file__).parent / "rtmpose_body.yaml"

_RTMPOSE_MEAN: tuple[float, float, float] = (123.675, 116.28, 103.53)
_RTMPOSE_STD: tuple[float, float, float] = (58.395, 57.12, 57.375)
_SIMCC_SPLIT_RATIO: float = 2.0

_MODEL_URLS: dict[str, str] = {
    "rtmpose-s_256x192": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip"
    ),
    "rtmpose-m_256x192": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
    ),
}

_INPUT_SIZES: dict[str, tuple[int, int]] = {
    "rtmpose-s_256x192": (256, 192),
    "rtmpose-m_256x192": (256, 192),
}

_POINT_NAMES: tuple[str, ...] = load_point_names(_YAML)
_NUM_KEYPOINTS = len(_POINT_NAMES)  # 23


class RTMPoseBodyDetectorConfig(KeypointDetectorConfig):
    detector_type: Literal["rtmpose_body"] = "rtmpose_body"
    session_backend: Literal["onnx"] = "onnx"
    model_name: str = "rtmpose-m_256x192"
    confidence_threshold: float = 0.004

    @property
    def input_size(self) -> tuple[int, int]:
        return _INPUT_SIZES.get(self.model_name, (256, 192))


@dataclass
class RTMPoseBodyDetector(KeypointDetector):
    """RTMPose body SIMCC detector — 23 keypoints (COCO17 + 6 foot points)."""

    config: RTMPoseBodyDetectorConfig
    session: OnnxSession
    _point_names: tuple[str, ...] = field(default_factory=lambda: _POINT_NAMES, init=False, repr=False)

    def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], RTMPoseMetadata]:
        """Letterbox-resize and normalise image for RTMPose inference.

        Returns (tensor, metadata) where tensor has shape (3, H, W) and dtype
        float32. metadata carries center/scale needed by postprocess to unproject
        SIMCC outputs back to image space.
        """
        h, w = image.shape[:2]
        crop_bbox = np.array([0.0, 0.0, float(w), float(h)], dtype=np.float64)
        input_h, input_w = self.config.input_size
        resized, center, scale = rtmpose_letterbox_preprocess(
            image, crop_bbox, (input_w, input_h), mean=_RTMPOSE_MEAN, std=_RTMPOSE_STD,
        )
        tensor = np.ascontiguousarray(resized.transpose(2, 0, 1).astype(np.float32))  # (3, H, W)
        return tensor, RTMPoseMetadata(center=center, scale=scale)

    def postprocess(self, raw: Any, metadata: RTMPoseMetadata) -> Keypoints:
        """Decode SIMCC outputs back to image-space keypoints.

        raw is [simcc_x, simcc_y] where each array has shape (1, N, bins)
        (already split from a batch — single-image slice along axis 0).
        """
        simcc_x, simcc_y = raw
        input_h, input_w = self.config.input_size
        keypoints_xy, scores = rtmpose_letterbox_postprocess(
            simcc_x=simcc_x,
            simcc_y=simcc_y,
            center=metadata.center,
            scale=metadata.scale,
            model_input_size=(input_w, input_h),
            simcc_split_ratio=_SIMCC_SPLIT_RATIO,
        )
        kpts_2d = keypoints_xy[0].copy()
        kpt_scores = scores[0]

        xyz = np.zeros((_NUM_KEYPOINTS, 3), dtype=np.float64)
        xyz[:, 0] = kpts_2d[:, 0]
        xyz[:, 1] = kpts_2d[:, 1]

        below_threshold = kpt_scores < self.config.confidence_threshold
        xyz[below_threshold] = np.nan

        visibility = kpt_scores.astype(np.float64)
        visibility[below_threshold] = 0.0

        return Keypoints(names=self._point_names, xyz=xyz, visibility=visibility)

    def detect(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext | None = None,
    ) -> Keypoints:
        img_h, img_w = image.shape[:2]
        crop_bbox = np.array([0.0, 0.0, float(img_w), float(img_h)], dtype=np.float64)

        input_h, input_w = self.config.input_size
        resized, center, scale = rtmpose_letterbox_preprocess(
            image, crop_bbox, (input_w, input_h), mean=_RTMPOSE_MEAN, std=_RTMPOSE_STD,
        )
        inp = np.ascontiguousarray(resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32))

        ort_session = self.session.get_session(self.config.model_name)
        input_name = ort_session.get_inputs()[0].name
        simcc_x, simcc_y = self.session.run(self.config.model_name, {input_name: inp})

        keypoints_xy, scores = rtmpose_letterbox_postprocess(
            simcc_x=simcc_x,
            simcc_y=simcc_y,
            center=center,
            scale=scale,
            model_input_size=(input_w, input_h),
            simcc_split_ratio=_SIMCC_SPLIT_RATIO,
        )
        kpts_2d = keypoints_xy[0].copy()
        kpt_scores = scores[0]

        xyz = np.zeros((_NUM_KEYPOINTS, 3), dtype=np.float64)
        xyz[:, 0] = kpts_2d[:, 0]
        xyz[:, 1] = kpts_2d[:, 1]

        below_threshold = kpt_scores < self.config.confidence_threshold
        xyz[below_threshold] = np.nan

        visibility = kpt_scores.astype(np.float64)
        visibility[below_threshold] = 0.0

        return Keypoints(names=self._point_names, xyz=xyz, visibility=visibility)

    @classmethod
    def connections(cls) -> tuple[tuple[str, str], ...]:
        return load_connections(_YAML)

    @classmethod
    def canonical_mapping_path(cls) -> Path:
        return Path(__file__).parent / "rtmpose_body_to_canonical_mapping.yaml"

    @classmethod
    def create(cls, config: KeypointDetectorConfig, session: Session) -> RTMPoseBodyDetector:
        if not isinstance(session, OnnxSession):
            raise TypeError(f"Expected OnnxSession, got {type(session).__name__}")
        if not isinstance(config, RTMPoseBodyDetectorConfig):
            raise TypeError(f"Expected RTMPoseBodyDetectorConfig, got {type(config).__name__}")
        return cls(config=config, session=session)

    @classmethod
    def model_spec(cls, model_name: str = "rtmpose-m_256x192") -> OnnxModelSpec:
        if model_name not in _INPUT_SIZES:
            raise ValueError(f"Unknown RTMPose body model {model_name!r}. Available: {list(_INPUT_SIZES)}")
        url = _MODEL_URLS.get(model_name)
        if url is None:
            raise ValueError(f"No download URL for RTMPose body model {model_name!r}.")
        return OnnxModelSpec(
            name=model_name,
            source=ModelSource(url=url),
            input_size=_INPUT_SIZES[model_name],
            prepare=None,
            coreml_options={"MLComputeUnits": "CPUAndGPU"},
        )


KEYPOINT_DETECTOR_REGISTRY["rtmpose_body"] = RTMPoseBodyDetector

RTMPOSE_BODY_MODEL_SPECS: dict[str, OnnxModelSpec] = {
    name: RTMPoseBodyDetector.model_spec(name) for name in _INPUT_SIZES
}
