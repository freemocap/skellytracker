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
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.config.detector_configs import ObjectDetectorConfig
from skellytracker.core.data_primitives import BoundingBox
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import (
    OBJECT_DETECTOR_REGISTRY,
    ObjectDetector,
)
from skellytracker.core.detectors.object_detectors.yolox._yolox_dynamic_batch import ensure_dynamic_batch, ensure_prenms_for_coreml
from skellytracker.core.detectors.object_detectors.yolox.yolox_preprocessing import multiclass_nms, yolox_letterbox_preprocess
from skellytracker.core.sessions.model_registry import ModelSource
from skellytracker.core.sessions.onnx_session import OnnxModelSpec, OnnxSession
from skellytracker.core.sessions.session import Session

logger = logging.getLogger(__name__)

_YOLOX_MODEL_URLS: dict[str, str] = {
    "yolox-tiny": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip"
    ),
    "yolox-m": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
    ),
}

# Canonical input sizes per model variant (H, W).
_YOLOX_INPUT_SIZES: dict[str, tuple[int, int]] = {
    "yolox-m": (640, 640),
    "yolox-tiny": (416, 416),
}


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
        return _YOLOX_INPUT_SIZES.get(self.model_name, (640, 640))


@dataclass
class YoloxPersonDetector(ObjectDetector):
    """YOLOX-based person detector.

    Stateless: no per-frame mutable state. Receives an OnnxSession at
    construction time; calls ``session.get_session(config.model_name)``
    on every ``detect()`` call.
    """

    config: YoloxPersonDetectorConfig
    session: OnnxSession

    def detect(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext | None = None,
    ) -> list[BoundingBox]:
        boxes, scores = _detect_yolox(
            image=image,
            session=self.session,
            model_name=self.config.model_name,
            input_size=self.config.input_size,
            score_threshold=self.config.score_threshold,
            nms_threshold=self.config.nms_threshold,
        )
        if len(boxes) == 0:
            return []

        result = [
            BoundingBox(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
                        confidence=float(s))
            for b, s in zip(boxes, scores, strict=False)
        ]
        result.sort(key=lambda bb: bb.confidence, reverse=True)

        if self.config.max_detections is not None:
            result = result[:self.config.max_detections]
        return result

    @classmethod
    def create(cls, config: ObjectDetectorConfig, session: Session) -> YoloxPersonDetector:
        if not isinstance(session, OnnxSession):
            raise TypeError(f"Expected OnnxSession, got {type(session).__name__}")
        if not isinstance(config, YoloxPersonDetectorConfig):
            raise TypeError(f"Expected YoloxPersonDetectorConfig, got {type(config).__name__}")
        return cls(config=config, session=session)

    @classmethod
    def connections(cls) -> tuple[tuple[str, str], ...]:
        return ()

    @classmethod
    def model_spec(cls, model_name: str = "yolox-m") -> OnnxModelSpec:
        """Return the OnnxModelSpec needed to load this model into an OnnxSession."""
        if model_name not in _YOLOX_INPUT_SIZES:
            raise ValueError(
                f"Unknown YOLOX model {model_name!r}. "
                f"Available: {list(_YOLOX_INPUT_SIZES)}"
            )
        url = _YOLOX_MODEL_URLS.get(model_name)
        if url is None:
            raise ValueError(f"No download URL registered for YOLOX model {model_name!r}.")
        return OnnxModelSpec(
            name=model_name,
            source=ModelSource(url=url),
            input_size=_YOLOX_INPUT_SIZES[model_name],
            prepare=ensure_dynamic_batch,
            coreml_prepare=ensure_prenms_for_coreml,
        )


OBJECT_DETECTOR_REGISTRY["yolox_person"] = YoloxPersonDetector

# Canonical model specs — include the one you need in OnnxSessionConfig.models.
YOLOX_MODEL_SPECS: dict[str, OnnxModelSpec] = {
    name: YoloxPersonDetector.model_spec(name) for name in _YOLOX_INPUT_SIZES
}


# ---------------------------------------------------------------------------
# Internal inference helpers
# ---------------------------------------------------------------------------

def _detect_yolox(
    image: NDArray[np.uint8],
    session: OnnxSession,
    model_name: str,
    input_size: tuple[int, int],
    score_threshold: float,
    nms_threshold: float,
) -> tuple[NDArray, NDArray]:
    """Run YOLOX on a single image.

    Returns
    -------
    boxes : (N, 4) float64  x1y1x2y2 in image pixel coords
    scores : (N,) float64
    """
    padded, ratio = yolox_letterbox_preprocess(image, input_size)
    inp = np.ascontiguousarray(
        padded.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    )
    input_name = session.get_session(model_name).get_inputs()[0].name
    outputs = session.run(model_name, {input_name: inp})

    if len(outputs) == 2:
        # Pre-NMS model (CoreML path): outputs are already-decoded xyxy boxes
        # and per-anchor confidence scores. Skip anchor decoding, apply threshold
        # and NMS directly.
        return _postprocess_prenms(
            boxes=outputs[0],
            scores=outputs[1],
            ratio=ratio,
            score_thr=score_threshold,
            nms_thr=nms_threshold,
        )
    return _postprocess_yolox(
        outputs_one=outputs[0],
        ratio=ratio,
        model_input_size=input_size,
        score_thr=score_threshold,
        nms_thr=nms_threshold,
    )


def _postprocess_prenms(
    boxes: NDArray,
    scores: NDArray,
    ratio: float,
    score_thr: float,
    nms_thr: float,
) -> tuple[NDArray, NDArray]:
    """Postprocess pre-NMS model outputs (CoreML path).

    The pre-NMS model skips the baked-in NMS subgraph and outputs all anchor
    predictions directly:
      boxes  : (N, 4)    already-decoded xyxy in letterboxed image space
      scores : (1, N)    per-anchor confidence (objectness × max class score)
    """
    if boxes.ndim == 3:
        boxes = boxes[0]  # (1, N, 4) → (N, 4)
    boxes = (boxes / ratio).astype(np.float64)
    scores = scores.reshape(-1).astype(np.float64)

    mask = scores > score_thr
    boxes, scores = boxes[mask], scores[mask]

    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)

    dets, _ = multiclass_nms(boxes, scores[:, None], nms_thr=nms_thr, score_thr=score_thr)
    if dets is None:
        return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)

    return dets[:, :4].astype(np.float64), dets[:, 4].astype(np.float64)


def _postprocess_yolox(
    outputs_one: NDArray,
    ratio: float,
    model_input_size: tuple[int, int],
    score_thr: float,
    nms_thr: float,
) -> tuple[NDArray, NDArray]:
    """Decode a single-image YOLOX output into (boxes, scores).

    Handles two output formats:
    - ``shape[-1] == 5``: NMS already applied by the ONNX graph;
      each row is [x1, y1, x2, y2, score].
    - ``shape[-1] == 4``: raw anchor grid output without NMS;
      applies grid/stride decoding + Python NMS.
    """
    if outputs_one.shape[-1] == 5:
        # NMS baked in: outputs_one is (1, num_dets, 5) = [x1,y1,x2,y2,score]
        boxes = outputs_one[0, :, :4] / ratio
        scores = outputs_one[0, :, 4]
        mask = scores > score_thr
        return boxes[mask].astype(np.float64), scores[mask].astype(np.float64)

    # Raw anchor output — need grid reconstruction + NMS.
    # outputs_one is (1, num_anchors, C) where C encodes cxcywh + objectness + cls.
    if outputs_one.shape[-1] == 4:
        # Stripped anchor format: cxcywh only — compute grids from model_input_size.
        grids: list[NDArray] = []
        expanded_strides: list[NDArray] = []
        strides = [8, 16, 32]
        hsizes = [model_input_size[0] // s for s in strides]
        wsizes = [model_input_size[1] // s for s in strides]
        for hsize, wsize, stride in zip(hsizes, wsizes, strides, strict=False):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
        grids_ = np.concatenate(grids, 1)
        expanded_strides_ = np.concatenate(expanded_strides, 1)
        outputs_one[..., :2] = (outputs_one[..., :2] + grids_) * expanded_strides_
        outputs_one[..., 2:4] = np.exp(outputs_one[..., 2:4]) * expanded_strides_

        predictions = outputs_one[0]
        boxes_cxcy = predictions[:, :4]
        obj_scores = predictions[:, 4:5]
        cls_scores = predictions[:, 5:]
        combined = obj_scores * cls_scores

        boxes_xyxy = np.empty_like(boxes_cxcy)
        boxes_xyxy[:, 0] = boxes_cxcy[:, 0] - boxes_cxcy[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes_cxcy[:, 1] - boxes_cxcy[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes_cxcy[:, 0] + boxes_cxcy[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes_cxcy[:, 1] + boxes_cxcy[:, 3] / 2.0
        boxes_xyxy /= ratio

        dets, _ = multiclass_nms(boxes_xyxy, combined, nms_thr=nms_thr, score_thr=score_thr)
        if dets is None:
            return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)

        final_boxes = dets[:, :4]
        final_scores = dets[:, 4]
        final_cls = dets[:, 5]
        mask = (final_scores > 0.3) & (final_cls == 0)
        return final_boxes[mask].astype(np.float64), final_scores[mask].astype(np.float64)

    raise RuntimeError(
        f"Unexpected YOLOX output shape: {outputs_one.shape}. "
        f"Expected last dim 4 or 5."
    )
