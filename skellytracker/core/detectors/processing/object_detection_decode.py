"""Generic object-detector decode utilities: NMS and box-format conversion.

Free functions any object detector's `postprocess()` can call — pure math
with no per-model-family assumptions, vendored from rtmlib
(``tools/object_detection/post_processings.py``). `decode_prenms`/
`decode_nms_baked_in` are configured by a `DetectionDecodeSpec` (the decode
contract a sidecar YAML describes), but that's just where the config value
comes from — this module is about decoding detections, not about the sidecar
format, so it lives with the other detector code rather than under
`core/sidecar/`.

`decode_prenms`/`decode_nms_baked_in` cover the two common ONNX export shapes
that need no further per-model decode math: pre-NMS boxes+scores with NMS
stripped from the graph, and rows that are already fully decoded and
NMS-filtered. A raw/undecoded anchor grid (e.g. YOLOX's stride-based head)
needs family-specific grid reconstruction and lives with that family's own
detector code instead — see
`skellytracker/core/detectors/object_detectors/yolox/yolox_decode.py`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.sidecar.model import DetectionDecodeSpec

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
# Box format conversion
# ==========================================================================


def boxes_to_xyxy(boxes: NDArray, box_format: str | None) -> NDArray:
    """Convert a `(..., 4)` box array from `box_format` into xyxy.

    Branches on every `decode.box_format` value the sidecar spec allows
    (`xyxy`/`xywh`/`cxcywh`); `None` is treated as `xyxy` (the spec default
    for a detector role).
    """
    fmt = box_format or "xyxy"
    if fmt == "xyxy":
        return boxes
    if fmt == "xywh":
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        return np.stack([x, y, x + w, y + h], axis=-1)
    if fmt == "cxcywh":
        cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        return np.stack(
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=-1
        )
    raise NotImplementedError(f"decode.box_format={fmt!r} is not yet implemented")


# ==========================================================================
# Generic decode strategies
# ==========================================================================


def decode_prenms(
    boxes: NDArray,
    scores: NDArray,
    ratio: float,
    score_thr: float,
    nms_thr: float,
    decode_spec: DetectionDecodeSpec,
) -> tuple[NDArray, NDArray]:
    """Postprocess pre-NMS model outputs for a single image.

    A pre-NMS model has the baked-in NMS subgraph stripped and outputs all
    anchor predictions directly:
      boxes  : (A, 4) or (1, A, 4)  already-decoded xyxy in letterboxed space
      scores : (A,) or (1, A)       per-anchor confidence (objectness × max class)

    ``A`` is the anchor count. Both arrays must describe exactly one image — a
    leading dim > 1 means a whole camera batch was passed in where a per-camera
    slice was expected, which would silently return camera 0's boxes for every
    camera.

    A stripped pre-NMS graph is only known to emit already-decoded xyxy rows
    — `decode.box_format` isn't threaded through this path yet, since a
    non-xyxy pre-NMS export hasn't been seen. Raise rather than silently
    mis-interpret the columns if one shows up.
    """
    if decode_spec.box_format not in (None, "xyxy"):
        raise NotImplementedError(
            "sidecar pre-NMS decode only supports box_format='xyxy', got "
            f"{decode_spec.box_format!r}"
        )
    if boxes.ndim == 3:
        if boxes.shape[0] != 1:
            raise ValueError(
                f"decode_prenms expects one image, got boxes with batch "
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
            f"decode_prenms expects one image, got scores with batch "
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


def decode_nms_baked_in(
    outputs_one: NDArray,
    ratio: float,
    score_thr: float,
    decode_spec: DetectionDecodeSpec,
) -> tuple[NDArray, NDArray]:
    """Decode a single-image detection tensor whose rows are already NMS-filtered.

    `outputs_one` is `(1, num_dets, 5)`: a box in `decode.box_format` plus a
    trailing score column. This is the generic "NMS baked into the graph"
    export shape — no per-model-family math needed beyond box-format
    conversion.
    """
    boxes = boxes_to_xyxy(outputs_one[0, :, :4], decode_spec.box_format) / ratio
    scores = outputs_one[0, :, 4]
    mask = scores > score_thr
    return boxes[mask].astype(np.float64), scores[mask].astype(np.float64)
