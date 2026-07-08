"""YOLOX-specific preprocessing and NMS utilities.

Vendored from rtmlib (``tools/object_detection/post_processings.py`` and
``tools/pose_estimation/rtmo.py``).  All functions are free functions —
no rtmlib class dependencies.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


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
                [valid_boxes[keep_indices], valid_scores[keep_indices, None], cls_inds], 1
            )
            final_dets.append(dets)
            last_keep = keep_indices
    if len(final_dets) == 0:
        return None, None
    final = np.concatenate(final_dets, 0)
    return final, last_keep
