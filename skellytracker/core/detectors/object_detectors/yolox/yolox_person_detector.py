"""YOLOX-based person detector — independent ObjectDetector implementation.

YOLOX is a general-purpose object detector used here to locate person bounding
boxes. It is completely decoupled from any keypoint estimation backend: any
ObjectDetector can be combined with any KeypointDetector in a DetectionStage.

The OnnxSession must be pre-loaded with the YOLOX model.  Use
``YoloxPersonDetector.model_spec(model_name)`` to get the OnnxModelSpec needed
by OnnxSessionConfig::

    session_config = OnnxSessionConfig(
        models=[YoloxPersonDetector.model_spec("yolox-m"), ...],
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.config.detector_configs import ObjectDetectorConfig
from skellytracker.core.data_primitives import BoundingBox
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import (
    OBJECT_DETECTOR_REGISTRY,
    ObjectDetector,
)
from skellytracker.core.detectors.processing.image_preprocessing import preprocess_image
from skellytracker.core.detectors.metadata import YoloxMetadata
from skellytracker.core.detectors.object_detectors.yolox._yolox_dynamic_batch import (
    prepare_yolox_onnx,
)
from skellytracker.core.detectors.object_detectors.yolox.yolox_decode import (
    yolox_detection_decode,
)
from skellytracker.core.sessions.onnx_session import OnnxModelSpec, OnnxSession
from skellytracker.core.sessions.session import Session
from skellytracker.core.sidecar.loader import load_sidecar
from skellytracker.core.sidecar.runtime import sidecar_model_spec

logger = logging.getLogger(__name__)

_SIDECAR_PATH = Path(__file__).parent / "yolox.yaml"
_SIDECAR = load_sidecar(_SIDECAR_PATH)


class YoloxPersonDetectorConfig(ObjectDetectorConfig):
    """Config for the YOLOX person detector.

    Attributes
    ----------
    model_name:
        Which YOLOX checkpoint to use.  Must match a key in
        ``YoloxPersonDetector.AVAILABLE_MODELS``.
    score_threshold:
        Minimum objectness score to keep a detection.
    nms_threshold:
        IoU threshold for non-maximum suppression.
    max_detections:
        Keep only the top-N detections by confidence. ``None`` = keep all.
    """

    detector_type: Literal["yolox_person"] = "yolox_person"
    session_backend: Literal["onnx"] = "onnx"
    model_name: str = "yolox-m"
    score_threshold: float = 0.7
    nms_threshold: float = 0.45
    max_detections: int | None = 1

    @property
    def input_size(self) -> tuple[int, int]:
        if self.model_name not in _SIDECAR.sizes:
            return (640, 640)
        resolved = _SIDECAR.resolved_size(self.model_name)
        target_size = resolved.input.resize.target_size
        return (target_size[0], target_size[1])


@dataclass
class YoloxPersonDetector(ObjectDetector):
    """YOLOX-based person detector.

    Stateless: no per-frame mutable state. Receives an OnnxSession at
    construction time; calls ``session.get_session(config.model_name)``
    on every ``detect()`` call.
    """

    config: YoloxPersonDetectorConfig
    session: OnnxSession

    def preprocess(
        self, image: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float32], YoloxMetadata]:
        """Letterbox-pad and transpose image for YOLOX inference.

        Returns (tensor, metadata) where tensor has shape (3, H, W) and dtype
        float32 — ready to stack into a batch. metadata.ratio is needed by
        postprocess to scale detections back to original image space.
        """
        tensor, ratio = preprocess_image(
            image, self.config.input_size, _SIDECAR.input, precision="fp32"
        )
        return tensor, YoloxMetadata(ratio=ratio)

    def postprocess(self, raw: Any, metadata: YoloxMetadata) -> list[BoundingBox]:
        """Decode per-image raw ORT outputs into BoundingBox list.

        raw is a list of arrays as returned by session.run for a single image
        (already split from a batch along axis 0). The number of outputs
        determines whether the model used pre-NMS or standard outputs.
        """
        if _SIDECAR.decode is None:
            raise ValueError(
                "yolox.yaml must declare `decode` (role includes object_detector)"
            )
        boxes_nd, scores_nd = yolox_detection_decode(
            raw,
            ratio=metadata.ratio,
            model_input_size=self.config.input_size,
            score_threshold=self.config.score_threshold,
            nms_threshold=self.config.nms_threshold,
            decode_spec=_SIDECAR.decode,
        )

        if len(boxes_nd) == 0:
            return []

        result = [
            BoundingBox(
                x1=float(b[0]),
                y1=float(b[1]),
                x2=float(b[2]),
                y2=float(b[3]),
                confidence=float(s),
            )
            for b, s in zip(boxes_nd, scores_nd, strict=False)
        ]
        result.sort(key=lambda bb: bb.confidence, reverse=True)
        if self.config.max_detections is not None:
            result = result[: self.config.max_detections]
        return result

    def detect(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext | None = None,
    ) -> list[BoundingBox]:
        tensor, metadata = self.preprocess(image)
        inp = np.ascontiguousarray(tensor[np.newaxis])
        input_name = (
            self.session.get_session(self.config.model_name).get_inputs()[0].name
        )
        outputs = self.session.run(self.config.model_name, {input_name: inp})
        return self.postprocess(outputs, metadata)

    @classmethod
    def create(
        cls, config: ObjectDetectorConfig, session: Session
    ) -> YoloxPersonDetector:
        if not isinstance(session, OnnxSession):
            raise TypeError(f"Expected OnnxSession, got {type(session).__name__}")
        if not isinstance(config, YoloxPersonDetectorConfig):
            raise TypeError(
                f"Expected YoloxPersonDetectorConfig, got {type(config).__name__}"
            )
        return cls(config=config, session=session)

    @classmethod
    def connections(cls) -> tuple[tuple[str, str], ...]:
        return ()

    @classmethod
    def model_spec(cls, model_name: str = "yolox-m") -> OnnxModelSpec:
        """Return the OnnxModelSpec needed to load this model into an OnnxSession."""
        if model_name not in _SIDECAR.sizes:
            raise ValueError(
                f"Unknown YOLOX model {model_name!r}. "
                f"Available: {list(_SIDECAR.sizes)}"
            )
        return sidecar_model_spec(
            _SIDECAR,
            size=model_name,
            batch_key="1",
            precision="fp32",
            name=model_name,
            sidecar_dir=_SIDECAR_PATH.parent,
            prepare=prepare_yolox_onnx,
        )


OBJECT_DETECTOR_REGISTRY["yolox_person"] = YoloxPersonDetector

# Canonical model specs — include the one you need in OnnxSessionConfig.models.
YOLOX_MODEL_SPECS: dict[str, OnnxModelSpec] = {
    name: YoloxPersonDetector.model_spec(name) for name in _SIDECAR.sizes
}
