"""RTMPose whole-body keypoint detector.

Estimates 133 named keypoints (COCO-WholeBody: body + hands + face) using
RTMPose SIMCC models via ONNX Runtime.  Completely decoupled from any person
detector — callers supply an optional BoundingBox from upstream detection, or
the full image is used as the crop region.

The OnnxSession must be pre-loaded with the RTMPose model.  Use
``RTMPoseKeypointDetector.model_spec(model_name)`` to get the OnnxModelSpec
needed by OnnxSessionConfig::

    session_config = OnnxSessionConfig(
        models=[RTMPoseKeypointDetector.model_spec("rtmw-x-l_256x192"), ...],
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from numpy.typing import NDArray

from skellytracker.core.config.detector_configs import KeypointDetectorConfig
from skellytracker.core.data_primitives import BoundingBox, Keypoints
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import (
    KEYPOINT_DETECTOR_REGISTRY,
    KeypointDetector,
)
from skellytracker.core.detectors.rtmpose.rtmpose_preprocessing import (
    rtmpose_letterbox_postprocess,
    rtmpose_letterbox_preprocess,
)
from skellytracker.core.sessions.model_registry import ModelSource
from skellytracker.core.sessions.onnx_session import OnnxModelSpec, OnnxSession
from skellytracker.core.sessions.session import Session

logger = logging.getLogger(__name__)

_SCHEMA_DIR = Path(__file__).parent / "names_and_connections"
_WHOLEBODY_YAML = _SCHEMA_DIR / "rtmpose_wholebody.yaml"

# Standard ImageNet-style BGR normalization used by all RTMPose/RTMDet models.
_RTMPOSE_MEAN: tuple[float, float, float] = (123.675, 116.28, 103.53)
_RTMPOSE_STD: tuple[float, float, float] = (58.395, 57.12, 57.375)
_SIMCC_SPLIT_RATIO: float = 2.0

_RTMPOSE_MODEL_URLS: dict[str, str] = {
    "rtmw-l-m_256x192": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
        "rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip"
    ),
    "rtmw-x-l_256x192": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
        "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip"
    ),
    "rtmw-x-l_384x288": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
        "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
    ),
}

# Available RTMPose wholebody models: model_name → (H, W) input size.
# Convention: (H, W) matches the model name suffix (e.g. 256x192 → H=256, W=192).
# The rtmlib preprocessing functions use (W, H) order internally — the detector
# swaps when calling them.
_RTMPOSE_INPUT_SIZES: dict[str, tuple[int, int]] = {
    "rtmw-x-l_384x288": (384, 288),
    "rtmw-x-l_256x192": (256, 192),
    "rtmw-l-m_256x192": (256, 192),
}


def _load_point_names(path: Path) -> tuple[str, ...]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return tuple(data["tracked_points"])


def _load_connections(path: Path) -> tuple[tuple[str, str], ...]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return tuple((a, b) for a, b in data.get("connections", []))


# Module-level cache so YAML is parsed once per process.
_POINT_NAMES: tuple[str, ...] = _load_point_names(_WHOLEBODY_YAML)
_NUM_KEYPOINTS = len(_POINT_NAMES)  # 133


class RTMPoseDetectorConfig(KeypointDetectorConfig):
    """Config for the RTMPose whole-body keypoint detector.

    Attributes
    ----------
    model_name:
        Which RTMPose checkpoint to use.  Must match a key in
        ``RTMPoseKeypointDetector.AVAILABLE_MODELS``.
    confidence_threshold:
        SIMCC softmax peak threshold.  Keypoints with a peak below this are
        treated as undetected (NaN coordinates, 0 visibility).
        Tuned for the SIMCC distribution — typical range 0.002–0.010.
    """

    detector_type: Literal["rtmpose"] = "rtmpose"
    session_backend: Literal["onnx"] = "onnx"
    model_name: str = "rtmw-x-l_256x192"
    confidence_threshold: float = 0.004

    @property
    def input_size(self) -> tuple[int, int]:
        return _RTMPOSE_INPUT_SIZES.get(self.model_name, (256, 192))


@dataclass
class RTMPoseKeypointDetector(KeypointDetector):
    """RTMPose whole-body SIMCC keypoint detector.

    Stateless: no per-frame mutable state. Receives an OnnxSession at
    construction time; calls ``session.get_session(config.model_name)``
    on every ``detect()`` call.

    Returns 133 named keypoints (COCO-WholeBody: 23 body + 21 right hand +
    21 left hand + 68 face).  Undetected points have NaN coordinates and
    0.0 visibility score.
    """

    config: RTMPoseDetectorConfig
    session: OnnxSession
    _point_names: tuple[str, ...] = field(default_factory=lambda: _POINT_NAMES, init=False, repr=False)

    def detect(
        self,
        image: NDArray[np.uint8],
        bbox: BoundingBox | None = None,
        context: DetectionContext | None = None,
    ) -> Keypoints:
        img_h, img_w = image.shape[:2]

        # DetectionStage already cropped the image to the bbox region before calling
        # this detector, so the image has its own (0,0) origin. Use the full crop
        # extent for preprocessing rather than the full-image bbox coordinates.
        # After postprocessing we translate back to full-image space.
        crop_bbox = np.array([0.0, 0.0, float(img_w), float(img_h)], dtype=np.float64)

        # input_size is (H, W). rtmlib preprocessing uses (W, H) internally.
        input_h, input_w = self.config.input_size
        input_size_wh = (input_w, input_h)

        resized, center, scale = rtmpose_letterbox_preprocess(
            image, crop_bbox, input_size_wh, mean=_RTMPOSE_MEAN, std=_RTMPOSE_STD,
        )
        inp = np.ascontiguousarray(resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32))

        ort_session = self.session.get_session(self.config.model_name)
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: inp})
        simcc_x, simcc_y = outputs[0], outputs[1]

        keypoints_xy, scores = rtmpose_letterbox_postprocess(
            simcc_x=simcc_x,
            simcc_y=simcc_y,
            center=center,
            scale=scale,
            model_input_size=input_size_wh,
            simcc_split_ratio=_SIMCC_SPLIT_RATIO,
        )
        # keypoints_xy: (1, K, 2) in crop-local pixel coords, scores: (1, K)
        kpts_2d = keypoints_xy[0].copy()  # (K, 2)
        kpt_scores = scores[0]            # (K,)

        # Translate from crop-local back to full-image coordinates.
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
        return _load_connections(_WHOLEBODY_YAML)

    @classmethod
    def create(cls, config: KeypointDetectorConfig, session: Session) -> RTMPoseKeypointDetector:
        if not isinstance(session, OnnxSession):
            raise TypeError(f"Expected OnnxSession, got {type(session).__name__}")
        if not isinstance(config, RTMPoseDetectorConfig):
            raise TypeError(f"Expected RTMPoseDetectorConfig, got {type(config).__name__}")
        return cls(config=config, session=session)

    @classmethod
    def model_spec(cls, model_name: str = "rtmw-x-l_256x192") -> OnnxModelSpec:
        """Return the OnnxModelSpec needed to load this model into an OnnxSession."""
        if model_name not in _RTMPOSE_INPUT_SIZES:
            raise ValueError(
                f"Unknown RTMPose model {model_name!r}. "
                f"Available: {list(_RTMPOSE_INPUT_SIZES)}"
            )
        url = _RTMPOSE_MODEL_URLS.get(model_name)
        if url is None:
            raise ValueError(f"No download URL registered for RTMPose model {model_name!r}.")
        return OnnxModelSpec(
            name=model_name,
            source=ModelSource(url=url),
            input_size=_RTMPOSE_INPUT_SIZES[model_name],
            prepare=None,
            # RTMPose's Neural Engine compilation fails (error -5) with default
            # CoreML options; CPUAndGPU routes to Metal and compiles cleanly.
            coreml_options={"MLComputeUnits": "CPUAndGPU"},
        )


KEYPOINT_DETECTOR_REGISTRY["rtmpose"] = RTMPoseKeypointDetector

# Canonical model specs — include the one you need in OnnxSessionConfig.models.
RTMPOSE_MODEL_SPECS: dict[str, OnnxModelSpec] = {
    name: RTMPoseKeypointDetector.model_spec(name) for name in _RTMPOSE_INPUT_SIZES
}
