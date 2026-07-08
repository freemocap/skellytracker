"""RTMPose single-hand keypoint detector (21 keypoints).

Detects keypoints for a single hand crop. The caller is responsible for
providing a tight hand bounding box (e.g. derived from wrist position of an
upstream body detector) and for tracking which hand (left/right) the crop
belongs to. Point names have no left/right prefix — add them at the stage
level when composing with a body detector.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.config.detector_configs import KeypointDetectorConfig
from skellytracker.core.data_primitives import BoundingBox, Keypoints
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import (
    KEYPOINT_DETECTOR_REGISTRY,
    KeypointDetector,
)
from skellytracker.core.detectors.rtmpose._schema_loader import load_connections, load_point_names
from skellytracker.core.detectors.rtmpose.rtmpose_preprocessing import (
    rtmpose_letterbox_postprocess,
    rtmpose_letterbox_preprocess,
)
from skellytracker.core.sessions.model_registry import ModelSource
from skellytracker.core.sessions.onnx_session import OnnxModelSpec, OnnxSession
from skellytracker.core.sessions.session import Session

logger = logging.getLogger(__name__)

_YAML = Path(__file__).parent / "rtmpose_hand.yaml"

_RTMPOSE_MEAN: tuple[float, float, float] = (123.675, 116.28, 103.53)
_RTMPOSE_STD: tuple[float, float, float] = (58.395, 57.12, 57.375)
_SIMCC_SPLIT_RATIO: float = 2.0

_MODEL_URLS: dict[str, str] = {
    "rtmpose-m_256x256": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
    ),
}

_INPUT_SIZES: dict[str, tuple[int, int]] = {
    "rtmpose-m_256x256": (256, 256),
}

_POINT_NAMES: tuple[str, ...] = load_point_names(_YAML)
_NUM_KEYPOINTS = len(_POINT_NAMES)  # 21


class RTMPoseHandDetectorConfig(KeypointDetectorConfig):
    detector_type: Literal["rtmpose_hand"] = "rtmpose_hand"
    session_backend: Literal["onnx"] = "onnx"
    model_name: str = "rtmpose-m_256x256"
    confidence_threshold: float = 0.004

    @property
    def input_size(self) -> tuple[int, int]:
        return _INPUT_SIZES.get(self.model_name, (256, 256))


@dataclass
class RTMPoseHandDetector(KeypointDetector):
    """RTMPose hand SIMCC detector — 21 keypoints for a single hand crop."""

    config: RTMPoseHandDetectorConfig
    session: OnnxSession
    _point_names: tuple[str, ...] = field(default_factory=lambda: _POINT_NAMES, init=False, repr=False)

    def detect(
        self,
        image: NDArray[np.uint8],
        bbox: BoundingBox | None = None,
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
        simcc_x, simcc_y = ort_session.run(None, {input_name: inp})

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

        if bbox is not None:
            kpts_2d[:, 0] += bbox.x1
            kpts_2d[:, 1] += bbox.y1

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
    def create(cls, config: KeypointDetectorConfig, session: Session) -> RTMPoseHandDetector:
        if not isinstance(session, OnnxSession):
            raise TypeError(f"Expected OnnxSession, got {type(session).__name__}")
        if not isinstance(config, RTMPoseHandDetectorConfig):
            raise TypeError(f"Expected RTMPoseHandDetectorConfig, got {type(config).__name__}")
        return cls(config=config, session=session)

    @classmethod
    def model_spec(cls, model_name: str = "rtmpose-m_256x256") -> OnnxModelSpec:
        if model_name not in _INPUT_SIZES:
            raise ValueError(f"Unknown RTMPose hand model {model_name!r}. Available: {list(_INPUT_SIZES)}")
        url = _MODEL_URLS.get(model_name)
        if url is None:
            raise ValueError(f"No download URL for RTMPose hand model {model_name!r}.")
        return OnnxModelSpec(
            name=model_name,
            source=ModelSource(url=url),
            input_size=_INPUT_SIZES[model_name],
            prepare=None,
            coreml_options={"MLComputeUnits": "CPUAndGPU"},
        )


KEYPOINT_DETECTOR_REGISTRY["rtmpose_hand"] = RTMPoseHandDetector

RTMPOSE_HAND_MODEL_SPECS: dict[str, OnnxModelSpec] = {
    name: RTMPoseHandDetector.model_spec(name) for name in _INPUT_SIZES
}
