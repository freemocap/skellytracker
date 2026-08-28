"""RTMPose face keypoint detector (106 keypoints, LaPa format).

Detects face landmarks from a face crop. The caller is responsible for
providing a tight face bounding box (e.g. derived from an upstream body or
face detector). Upstream face detection is recommended for accuracy.
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
from skellytracker.core.detectors.processing.image_preprocessing import preprocess_image
from skellytracker.core.detectors.processing.simcc_decode import simcc_pose_decode
from skellytracker.core.detectors.metadata import RTMPoseMetadata
from skellytracker.core.sessions.onnx_session import OnnxModelSpec, OnnxSession
from skellytracker.core.sessions.session import Session
from skellytracker.core.sidecar.loader import load_sidecar
from skellytracker.core.sidecar.runtime import sidecar_model_spec

logger = logging.getLogger(__name__)

_SIDECAR_PATH = Path(__file__).parent / "rtmpose-face.yaml"
_SIDECAR = load_sidecar(_SIDECAR_PATH)

_POINT_NAMES: tuple[str, ...] = tuple(_SIDECAR.pose.tracked_points)
_NUM_KEYPOINTS = len(_POINT_NAMES)  # 106
_CONNECTIONS: tuple[tuple[str, str], ...] = tuple(
    tuple(edge) for edge in _SIDECAR.pose.connections[0].edges
)


class RTMPoseFaceDetectorConfig(KeypointDetectorConfig):
    detector_type: Literal["rtmpose_face"] = "rtmpose_face"
    session_backend: Literal["onnx"] = "onnx"
    model_name: str = "rtmpose-m_256x256"
    confidence_threshold: float = 0.004

    @property
    def input_size(self) -> tuple[int, int]:
        if self.model_name not in _SIDECAR.sizes:
            return (256, 256)
        resolved = _SIDECAR.resolved_size(self.model_name)
        target_size = resolved.input.resize.target_size
        return (target_size[0], target_size[1])


@dataclass
class RTMPoseFaceDetector(KeypointDetector):
    """RTMPose face SIMCC detector — 106 keypoints (LaPa format)."""

    config: RTMPoseFaceDetectorConfig
    session: OnnxSession
    _point_names: tuple[str, ...] = field(
        default_factory=lambda: _POINT_NAMES, init=False, repr=False
    )

    def preprocess(
        self, image: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float32], RTMPoseMetadata]:
        """Affine-crop and normalize image for RTMPose inference.

        Returns (tensor, metadata) where tensor has shape (3, H, W) and dtype
        float32. metadata carries center/scale needed by postprocess to
        unproject SIMCC outputs back to image space.
        """
        tensor, resize_meta = preprocess_image(
            image, self.config.input_size, _SIDECAR.input, precision="fp32"
        )
        center, scale = resize_meta  # affine_person_crop always returns (center, scale)
        return tensor, RTMPoseMetadata(center=center, scale=scale)

    def postprocess(self, raw: Any, metadata: RTMPoseMetadata) -> Keypoints:
        """Decode SIMCC outputs back to image-space keypoints.

        raw is [simcc_x, simcc_y] where each array has shape (1, N, bins)
        (already split from a batch — single-image slice along axis 0).
        """
        input_h, input_w = self.config.input_size
        kpts_2d, kpt_scores = simcc_pose_decode(
            raw=raw,
            center=metadata.center,
            scale=metadata.scale,
            model_input_size=(input_w, input_h),
            decode_spec=_SIDECAR.pose.decode,
        )

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
        tensor, metadata = self.preprocess(image)
        inp = np.ascontiguousarray(tensor[np.newaxis])

        ort_session = self.session.get_session(self.config.model_name)
        input_name = ort_session.get_inputs()[0].name
        raw = self.session.run(self.config.model_name, {input_name: inp})

        return self.postprocess(raw, metadata)

    @classmethod
    def connections(cls) -> tuple[tuple[str, str], ...]:
        return _CONNECTIONS

    @classmethod
    def create(
        cls, config: KeypointDetectorConfig, session: Session
    ) -> RTMPoseFaceDetector:
        if not isinstance(session, OnnxSession):
            raise TypeError(f"Expected OnnxSession, got {type(session).__name__}")
        if not isinstance(config, RTMPoseFaceDetectorConfig):
            raise TypeError(
                f"Expected RTMPoseFaceDetectorConfig, got {type(config).__name__}"
            )
        return cls(config=config, session=session)

    @classmethod
    def model_spec(cls, model_name: str = "rtmpose-m_256x256") -> OnnxModelSpec:
        if model_name not in _SIDECAR.sizes:
            raise ValueError(
                f"Unknown RTMPose face model {model_name!r}. Available: {list(_SIDECAR.sizes)}"
            )
        return sidecar_model_spec(
            _SIDECAR,
            size=model_name,
            batch_key="1",
            precision="fp32",
            name=model_name,
            sidecar_dir=_SIDECAR_PATH.parent,
            prepare=None,
            coreml_options={"MLComputeUnits": "CPUAndGPU"},
        )


KEYPOINT_DETECTOR_REGISTRY["rtmpose_face"] = RTMPoseFaceDetector

RTMPOSE_FACE_MODEL_SPECS: dict[str, OnnxModelSpec] = {
    name: RTMPoseFaceDetector.model_spec(name) for name in _SIDECAR.sizes
}
