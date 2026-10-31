"""RTMPose whole-body keypoint detector (133 keypoints: body + hands + face).

Models use COCO-WholeBody SIMCC outputs via ONNX Runtime. Callers supply an
optional BoundingBox crop; the full image is used if none is provided.
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
from skellytracker.core.detectors.keypoint_detectors._schema_loader import (
    load_connections,
    load_point_names,
)
from skellytracker.core.detectors.metadata import RTMPoseMetadata
from skellytracker.core.detectors.overlay_groups import resolve_overlay_groups
from skellytracker.core.sessions.onnx_model_spec import OnnxModelSpec
from skellytracker.core.sessions.session import Session

logger = logging.getLogger(__name__)

_YAML = Path(__file__).parents[2] / "rtmpose" / "wholebody" / "rtmpose_wholebody.yaml"

from skellytracker.core.sidecar.loader import load_sidecar
from skellytracker.core.sidecar.runtime import sidecar_model_spec
from skellytracker.core.detectors.processing.image_preprocessing import (
    preprocess_image,
    build_normalization_fn,
)
from skellytracker.core.detectors.processing.simcc_decode import simcc_pose_decode

_SIDECAR_PATH = Path(__file__).parent / "rtmw-wholebody.yaml"
_SIDECAR = load_sidecar(_SIDECAR_PATH)
_INPUT_SIZES = {
    name: _SIDECAR.resolved_size(name).input.resize.target_size
    for name in _SIDECAR.sizes
}
_NORMALIZE_IMAGE = build_normalization_fn(_SIDECAR.input, "fp32")
# Public point order remains the established application contract.
_POINT_NAMES = load_point_names(_YAML)
_NUM_KEYPOINTS = len(_POINT_NAMES)  # 133
_NATIVE_INDEX = {name: index for index, name in enumerate(_SIDECAR.pose.tracked_points)}
if set(_NATIVE_INDEX) != set(_POINT_NAMES):
    raise ValueError("RTMW sidecar and public landmark identities differ")
_MODEL_TO_SCHEMA_PERM = np.asarray(
    [_NATIVE_INDEX[name] for name in _POINT_NAMES], dtype=np.intp
)
_MODEL_TO_SCHEMA_PERM.flags.writeable = False


class RTMWWholebodyDetectorConfig(KeypointDetectorConfig):
    detector_type: Literal["rtmpose"] = "rtmpose"
    session_backend: Literal["onnx"] = "onnx"
    model_name: str = "rtmw-x-l_256x192"
    # Raw SIMCC peak response, clipped to [0, 1] — see get_simcc_maximum.
    #
    # Measured on rtmw-x-l_256x192 against real recordings, the response falls
    # into three bands:
    #   <= 0.33  person-free crops (floor, ceiling, wall, furniture)
    #   <= 0.36  synthetic blank / uniform-noise images
    #   0.42-0.57  heavily occluded joints, inferred from surrounding context
    #   0.62-1.00  clearly visible joints (per-group p10)
    #
    # 0.4 is the first value clear of the no-person band, so "no person" decodes
    # to no keypoints, while still keeping >=96% of visible keypoints in every
    # group. Note this is above mmpose/rtmlib's kpt_thr=0.3 convention, which
    # for this model sits *inside* the noise band and lets blank frames through.
    #
    # Deliberately no stricter than that: this NaNs coordinates outright, and
    # multi-view triangulation rejects bad points by reprojection error far
    # better than one view's confidence can. Raise toward 0.5 for aggressive 2D
    # gating — on an occluded-legs A/B, 0.4 separates occluded from visible by
    # 28pp and 0.5 by 62pp, at the cost of dropping more genuine detections.
    confidence_threshold: float = 0.4

    @property
    def input_size(self) -> tuple[int, int]:
        return _INPUT_SIZES[self.model_name]


@dataclass
class RTMWWholebodyDetector(KeypointDetector):
    """RTMPose whole-body SIMCC detector — 133 keypoints (body + hands + face)."""

    config: RTMWWholebodyDetectorConfig
    session: Session
    _point_names: tuple[str, ...] = field(
        default_factory=lambda: _POINT_NAMES, init=False, repr=False
    )

    def preprocess(
        self, image: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float32], RTMPoseMetadata]:
        tensor, (center, scale) = preprocess_image(
            image,
            self.config.input_size,
            _SIDECAR.input,
            precision="fp32",
            normalization_fn=_NORMALIZE_IMAGE,
        )
        return tensor, RTMPoseMetadata(center=center, scale=scale)

    def postprocess(self, raw: Any, metadata: RTMPoseMetadata) -> Keypoints:
        if len(raw) != 2 or any(
            output.ndim != 3 or output.shape[1] != _NUM_KEYPOINTS for output in raw
        ):
            raise ValueError("RTMW outputs disagree with the declared landmark count")
        input_h, input_w = self.config.input_size
        native_xy, native_scores = simcc_pose_decode(
            raw,
            metadata.center,
            metadata.scale,
            (input_w, input_h),
            _SIDECAR.pose.decode,
        )
        point_xy = native_xy[_MODEL_TO_SCHEMA_PERM]
        scores = native_scores[_MODEL_TO_SCHEMA_PERM]
        xyz = np.zeros((_NUM_KEYPOINTS, 3), dtype=np.float64)
        xyz[:, :2] = point_xy
        unavailable = scores < self.config.confidence_threshold
        xyz[unavailable] = np.nan
        visibility = scores.astype(np.float64)
        visibility[unavailable] = 0.0
        return Keypoints(names=self._point_names, xyz=xyz, visibility=visibility)

    def detect(
        self, image: NDArray[np.uint8], context: DetectionContext | None = None
    ) -> Keypoints:
        tensor, metadata = self.preprocess(image)
        input_name = (
            self.session.get_session(self.config.model_name).get_inputs()[0].name
        )
        raw = self.session.run(self.config.model_name, {input_name: tensor[np.newaxis]})
        return self.postprocess(raw, metadata)

    @classmethod
    def connections(cls) -> tuple[tuple[str, str], ...]:
        return load_connections(_YAML)

    @classmethod
    def connection_groups(cls) -> dict[str, tuple[tuple[str, str], ...]]:
        return resolve_overlay_groups(_SIDECAR.overlay, cls.connections())

    @classmethod
    def create(
        cls, config: KeypointDetectorConfig, session: Session
    ) -> RTMWWholebodyDetector:
        from skellytracker.core.sessions.onnx_session import OnnxSession  # noqa: PLC0415 — heavy; deferred

        if not isinstance(session, OnnxSession):
            raise TypeError(f"Expected OnnxSession, got {type(session).__name__}")
        if not isinstance(config, RTMWWholebodyDetectorConfig):
            raise TypeError(
                f"Expected RTMWWholebodyDetectorConfig, got {type(config).__name__}"
            )
        return cls(config=config, session=session)

    @classmethod
    def model_spec(cls, model_name: str = "rtmw-x-l_256x192") -> OnnxModelSpec:
        if model_name not in _INPUT_SIZES:
            raise ValueError(f"Unknown RTMPose model {model_name!r}")
        return sidecar_model_spec(
            _SIDECAR,
            size=model_name,
            batch_key="1",
            precision="fp32",
            name=model_name,
            sidecar_dir=_SIDECAR_PATH.parent,
            coreml_options={"MLComputeUnits": "CPUAndGPU"},
        )


KEYPOINT_DETECTOR_REGISTRY["rtmpose"] = RTMWWholebodyDetector

RTMW_WHOLEBODY_MODEL_SPECS: dict[str, OnnxModelSpec] = {
    name: RTMWWholebodyDetector.model_spec(name) for name in _INPUT_SIZES
}
