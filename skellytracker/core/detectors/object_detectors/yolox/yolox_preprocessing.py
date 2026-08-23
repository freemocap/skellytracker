"""YOLOX-specific preprocessing and NMS utilities.

Vendored from rtmlib (``tools/object_detection/post_processings.py`` and
``tools/pose_estimation/rtmo.py``).  All functions are free functions —
no rtmlib class dependencies.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from skellytracker.core.sidecar.model import DetectionDecodeSpec, InputSpec, Precision
from skellytracker.core.sidecar.runtime import build_normalization_fn

# ==========================================================================
# YOLOX detection preprocessing
# ==========================================================================


def yolox_letterbox_preprocess(
    img: NDArray[np.uint8],
    model_input_size: tuple[int, int],
) -> tuple[NDArray, float]:
    """YOLOX letterbox — resize to fit, pad with gray, no normalisation.

    Returns ``(padded_img, ratio)`` where *padded_img* is uint8 (not float32
    — the YOLOX ONNX graph handles input normalisation internally).
    """
    th, tw = model_input_size
    padded_img = np.full((th, tw, 3), 114, dtype=np.uint8)

    ratio = min(th / img.shape[0], tw / img.shape[1])
    nw, nh = int(img.shape[1] * ratio), int(img.shape[0] * ratio)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    padded_img[:nh, :nw] = resized

    return padded_img, ratio


# ==========================================================================
# NMS utilities
# ==========================================================================


def nms(
    boxes: NDArray,
    scores: NDArray,
    nms_thr: float,
) -> list[int]:
    """Single-class NMS implemented in numpy.

    Parameters
    ----------
    boxes : np.ndarray  shape (N, 4)  x1y1x2y2.
    scores : np.ndarray  shape (N,)
    nms_thr : float  IoU threshold.

    Returns
    -------
    keep : list[int]  indices of boxes to keep.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(
    boxes: NDArray,
    scores: NDArray,
    nms_thr: float,
    score_thr: float,
) -> tuple[NDArray | None, list[int] | None]:
    """Multiclass NMS — class-aware version.

    Parameters
    ----------
    boxes : np.ndarray  shape (N, 4)  x1y1x2y2.
    scores : np.ndarray  shape (N, C)
    nms_thr : float
    score_thr : float

    Returns
    -------
    dets : np.ndarray | None  shape (M, 6)  [x1, y1, x2, y2, score, class]
    keep : list[int] | None
    """
    final_dets: list[NDArray] = []
    last_keep: list[int] | None = None
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        keep_indices = nms(valid_boxes, valid_scores, nms_thr)
        if len(keep_indices) > 0:
            cls_inds = np.ones((len(keep_indices), 1)) * cls_ind
            dets = np.concatenate(
                [valid_boxes[keep_indices], valid_scores[keep_indices, None], cls_inds],
                1,
            )
            final_dets.append(dets)
            last_keep = keep_indices
    if len(final_dets) == 0:
        return None, None
    final = np.concatenate(final_dets, 0)
    return final, last_keep


# ==========================================================================
# YOLOX output decoding
# ==========================================================================


def _postprocess_prenms(
    boxes: NDArray,
    scores: NDArray,
    ratio: float,
    score_thr: float,
    nms_thr: float,
) -> tuple[NDArray, NDArray]:
    """Postprocess pre-NMS model outputs for a single image.

    The pre-NMS model has the baked-in NMS subgraph stripped and outputs all
    anchor predictions directly:
      boxes  : (A, 4) or (1, A, 4)  already-decoded xyxy in letterboxed space
      scores : (A,) or (1, A)       per-anchor confidence (objectness × max class)

    ``A`` is the anchor count. Both arrays must describe exactly one image — a
    leading dim > 1 means a whole camera batch was passed in where a per-camera
    slice was expected, which would silently return camera 0's boxes for every
    camera.
    """
    if boxes.ndim == 3:
        if boxes.shape[0] != 1:
            raise ValueError(
                f"_postprocess_prenms expects one image, got boxes with batch "
                f"dim {boxes.shape[0]} (shape {boxes.shape}). Split the batch "
                f"per camera before calling postprocess."
            )
        boxes = boxes[0]
    elif boxes.ndim != 2:
        raise ValueError(
            f"Expected pre-NMS boxes with ndim 2 or 3, got shape {boxes.shape}."
        )

    if scores.ndim > 1 and scores.shape[0] != 1:
        raise ValueError(
            f"_postprocess_prenms expects one image, got scores with batch "
            f"dim {scores.shape[0]} (shape {scores.shape}). Split the batch "
            f"per camera before calling postprocess."
        )

    boxes = (boxes / ratio).astype(np.float64)
    scores = scores.reshape(-1).astype(np.float64)

    if scores.shape[0] != boxes.shape[0]:
        raise ValueError(
            f"Pre-NMS anchor count mismatch: {boxes.shape[0]} boxes vs "
            f"{scores.shape[0]} scores. The two pre-NMS graph outputs are not "
            f"aligned to the same anchor grid."
        )

    mask = scores > score_thr
    boxes, scores = boxes[mask], scores[mask]

    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)

    dets, _ = multiclass_nms(
        boxes, scores[:, None], nms_thr=nms_thr, score_thr=score_thr
    )
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

        dets, _ = multiclass_nms(
            boxes_xyxy, combined, nms_thr=nms_thr, score_thr=score_thr
        )
        if dets is None:
            return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)

        final_boxes = dets[:, :4]
        final_scores = dets[:, 4]
        final_cls = dets[:, 5]
        mask = (final_scores > 0.3) & (final_cls == 0)
        return final_boxes[mask].astype(np.float64), final_scores[mask].astype(
            np.float64
        )

    raise RuntimeError(
        f"Unexpected YOLOX output shape: {outputs_one.shape}. "
        f"Expected last dim 4 or 5."
    )


# ==========================================================================
# Sidecar-driven dispatch wrappers
# ==========================================================================


def _check_letterbox_resize_spec(input_spec: InputSpec) -> None:
    """Guard the subset of `input.resize` this module actually implements.

    `yolox_letterbox_preprocess` always does: method=letterbox, pad with 114,
    preserve aspect ratio, `cv2.INTER_LINEAR`. It does not read `input_spec.resize`
    at all, so a sidecar declaring anything else (e.g. `affine_person_crop`, a
    different pad value, `preserve_aspect_ratio: false`) would silently get the
    wrong preprocessing. Fail loudly instead until this module grows support for
    those cases.
    """
    resize = input_spec.resize
    if resize is None or resize.method != "letterbox":
        raise ValueError(
            f"sidecar_letterbox_preprocess only implements resize.method='letterbox', "
            f"got {resize.method if resize else None!r}"
        )
    if resize.pad_value not in (None, 114):
        raise ValueError(
            f"sidecar_letterbox_preprocess hardcodes pad_value=114, got {resize.pad_value!r}"
        )
    if resize.preserve_aspect_ratio not in (None, True):
        raise ValueError(
            "sidecar_letterbox_preprocess always preserves aspect ratio, got "
            f"preserve_aspect_ratio={resize.preserve_aspect_ratio!r}"
        )
    if resize.interpolation != "linear":
        raise ValueError(
            f"sidecar_letterbox_preprocess hardcodes cv2.INTER_LINEAR, got "
            f"interpolation={resize.interpolation!r}"
        )


def sidecar_letterbox_preprocess(
    image: NDArray[np.uint8],
    target_size: tuple[int, int],
    input_spec: InputSpec,
    precision: Precision = "fp32",
) -> tuple[NDArray[np.float32], float]:
    """Letterbox `image` to `target_size` and apply `input_spec`'s normalization.

    Thin wrapper around `yolox_letterbox_preprocess`; the letterbox math itself
    is in `yolox_letterbox_preprocess`. Returns `(tensor, ratio)` where `tensor` is laid out per
    `input_spec.layout` (`NCHW` transposes; `NHWC` leaves HWC as-is).

    Only supports the `input.resize` variant YOLOX's sidecar actually declares
    (letterbox, pad_value 114, aspect-preserving, linear interpolation) — see
    `_check_letterbox_resize_spec`.
    """
    _check_letterbox_resize_spec(input_spec)
    padded, ratio = yolox_letterbox_preprocess(image, target_size)
    normalize = build_normalization_fn(input_spec, precision)
    normalized = normalize(padded)
    tensor = (
        normalized if input_spec.layout == "NHWC" else normalized.transpose(2, 0, 1)
    )
    return np.ascontiguousarray(tensor.astype(np.float32)), ratio


def _check_yolox_decode_spec(decode_spec: DetectionDecodeSpec) -> None:
    """Guard the subset of `decode` this module actually implements.

    `_postprocess_prenms`/`_postprocess_yolox` always: return `xyxy` boxes,
    treat class 0 as the only class kept (person), and read score as a single
    trailing column. Neither function reads `decode` at all, so a sidecar
    declaring a different `box_format`/`person_class_id`/`class_id_base` would
    silently get boxes decoded under the wrong assumptions. Fail loudly instead
    until this module grows support for those cases.
    """
    if decode_spec.box_format not in (None, "xyxy"):
        raise ValueError(
            f"sidecar_detection_decode only produces box_format='xyxy', got "
            f"{decode_spec.box_format!r}"
        )
    if decode_spec.score_field not in (None, "score"):
        raise ValueError(
            f"sidecar_detection_decode hardcodes score_field='score', got "
            f"{decode_spec.score_field!r}"
        )
    if decode_spec.class_id_base != 0:
        raise ValueError(
            f"sidecar_detection_decode assumes class_id_base=0, got "
            f"{decode_spec.class_id_base!r}"
        )
    if decode_spec.person_class_id != 0:
        raise ValueError(
            f"sidecar_detection_decode hardcodes person_class_id=0 (see "
            f"_postprocess_yolox's `final_cls == 0` filter), got "
            f"{decode_spec.person_class_id!r}"
        )


def sidecar_detection_decode(
    raw: list[NDArray],
    ratio: float,
    model_input_size: tuple[int, int],
    score_threshold: float,
    nms_threshold: float,
    decode_spec: DetectionDecodeSpec,
) -> tuple[NDArray, NDArray]:
    """Decode raw YOLOX ONNX outputs for a single image into `(boxes, scores)`.

    Dispatches on `len(raw)` exactly as the pre-sidecar `YoloxPersonDetector`
    did: 2 outputs means the ONNX graph's baked-in NMS was stripped (pre-NMS
    boxes + scores); otherwise a single output tensor carries either
    NMS-baked-in `[x1,y1,x2,y2,score]` rows or a raw anchor grid, both handled
    by `_postprocess_yolox`. Decode math is unchanged from the pre-migration
    implementation.

    Only supports the `decode` variant YOLOX's sidecar actually declares (xyxy
    boxes, class 0 = person) — see `_check_yolox_decode_spec`.
    """
    _check_yolox_decode_spec(decode_spec)
    if len(raw) == 2:
        return _postprocess_prenms(
            boxes=raw[0],
            scores=raw[1],
            ratio=ratio,
            score_thr=score_threshold,
            nms_thr=nms_threshold,
        )
    return _postprocess_yolox(
        outputs_one=raw[0],
        ratio=ratio,
        model_input_size=model_input_size,
        score_thr=score_threshold,
        nms_thr=nms_threshold,
    )
