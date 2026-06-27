"""Vendored post-processing functions originally from rtmlib.

Pure numpy — no rtmlib internals.  These are the only functions from rtmlib's
``tools/pose_estimation/post_processings.py`` and
``tools/object_detection/post_processings.py`` that skellytracker uses.
"""

import numpy as np
from numpy.typing import NDArray


def _stable_softmax(x: NDArray) -> NDArray:
    """Softmax with log-sum-exp trick for numerical stability.

    Parameters
    ----------
    x : np.ndarray  shape (..., D)

    Returns
    -------
    np.ndarray  same shape, softmax along last axis.
    """
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def get_simcc_maximum(
    simcc_x: NDArray,
    simcc_y: NDArray,
) -> tuple[NDArray, NDArray]:
    """Decode SIMCC heatmaps to (x, y) coordinates with confidence scores.

    Applies softmax to raw logits, then returns the peak probability as
    confidence.  Value is in [0, 1]: higher = more mass concentrated in
    the winning bin.  A uniform distribution over N bins gives ~1/N.

    Parameters
    ----------
    simcc_x : np.ndarray  shape (N, K, Wx) — raw SIMCC logits for x-axis.
    simcc_y : np.ndarray  shape (N, K, Wy) — raw SIMCC logits for y-axis.

    Returns
    -------
    locs : np.ndarray  shape (N, K, 2)  x/y keypoint coordinates.
    vals : np.ndarray  shape (N, K)     confidence = average peak softmax
                                        probability across x and y axes.
    """
    N, K, Wx = simcc_x.shape
    Wy = simcc_y.shape[-1]
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # Convert raw logits → probability distributions.
    px = _stable_softmax(simcc_x)
    py = _stable_softmax(simcc_y)

    # Argmax is invariant under softmax (monotonic).
    x_locs = np.argmax(px, axis=1)
    y_locs = np.argmax(py, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)

    max_px = np.amax(px, axis=1)
    max_py = np.amax(py, axis=1)

    vals = 0.5 * (max_px + max_py)
    locs[vals <= 0.0] = -1

    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


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


def convert_coco_to_openpose(
    keypoints: NDArray,
    scores: NDArray,
) -> tuple[NDArray, NDArray]:
    """Reorder COCO-17 keypoints to OpenPose 18-keypoint convention.

    Computes neck as the midpoint of left/right shoulders and inserts it
    at index 1, then permutes the remaining points.
    """
    keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)

    # neck = midpoint of left-shoulder (5) and right-shoulder (6)
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
    neck[:, 2:3] = np.where(
        keypoints_info[:, 5, 2:3] > keypoints_info[:, 6, 2:3],
        keypoints_info[:, 6, 2:3],
        keypoints_info[:, 5, 2:3],
    )
    new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
    keypoints_info = new_keypoints_info

    keypoints_out = keypoints_info[..., :2]
    scores_out = keypoints_info[..., 2]
    return keypoints_out, scores_out
