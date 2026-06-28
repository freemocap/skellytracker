"""Vendored preprocessing functions originally from rtmlib.

Extracted from ``rtmlib/tools/pose_estimation/rtmo.py`` and
``rtmlib/tools/pose_estimation/rtmpose.py`` (plus the affine helpers in
``rtmlib/tools/pose_estimation/pre_processings.py``).

All functions are free functions — no rtmlib class dependencies.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from skellytracker.trackers.old.utilities.gpu_utils.rtm_postprocessing import (
    get_simcc_maximum,
    multiclass_nms,
)

# ==========================================================================
# RTMO (one-stage body) preprocessing / postprocessing
# ==========================================================================


def rtmo_preprocess(
    img: NDArray[np.uint8],
    model_input_size: tuple[int, int],
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> tuple[NDArray, float]:
    """Letterbox + normalise for RTMO one-stage body model.

    Parameters
    ----------
    img : np.ndarray  H×W×3 BGR uint8
    model_input_size : (H, W)  e.g. (640, 640)
    mean : optional BGR mean for normalisation.
    std : optional BGR std for normalisation.

    Returns
    -------
    padded_img : np.ndarray  (H, W, 3) float32
    ratio : float  scale factor (original → model input space).
    """
    th, tw = model_input_size
    padded_img = np.full((th, tw, 3), 114, dtype=np.float32)

    ratio = min(th / img.shape[0], tw / img.shape[1])
    nw, nh = int(img.shape[1] * ratio), int(img.shape[0] * ratio)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    padded_img[:nh, :nw] = resized

    if mean is not None and std is not None:
        padded_img = (padded_img - np.array(mean, dtype=np.float32)) / np.array(
            std, dtype=np.float32
        )

    return padded_img, ratio


def rtmo_postprocess(
    outputs: list[NDArray],
    ratio: float = 1.0,
    nms_thr: float = 0.45,
    score_thr: float = 0.7,
) -> tuple[NDArray, NDArray]:
    """Decode RTMO model outputs → keypoints + scores.

    Parameters
    ----------
    outputs : [det_outputs, pose_outputs] from ONNX session.run()
        det_outputs  — (1, N_det, 5)  [x1, y1, x2, y2, score]
        pose_outputs — (1, N_det, K, 3)  [x, y, score]
    ratio : letterbox scale factor.
    nms_thr : NMS IoU threshold.
    score_thr : NMS score threshold.

    Returns
    -------
    keypoints : np.ndarray  (M, 17, 2)
    scores : np.ndarray     (M, 17)
    """
    det_outputs, pose_outputs = outputs

    final_boxes = det_outputs[0, :, :4]
    final_scores = det_outputs[0, :, 4]
    final_boxes = final_boxes / ratio

    keypoints = pose_outputs[0, :, :, :2]
    scores = pose_outputs[0, :, :, 2]
    keypoints = keypoints / ratio

    dets, keep = multiclass_nms(
        final_boxes,
        final_scores[:, np.newaxis],
        nms_thr=nms_thr,
        score_thr=score_thr,
    )
    if keep is not None:
        keypoints = keypoints[keep]
        scores = scores[keep]
    else:
        keypoints = np.expand_dims(np.zeros_like(keypoints[0]), axis=0)
        scores = np.expand_dims(np.zeros_like(scores[0]), axis=0)

    return keypoints, scores


# ==========================================================================
# Affine preprocessing helpers (from rtmlib's pre_processings.py)
# ==========================================================================


def _rotate_point(pt: NDArray, angle_rad: float) -> NDArray:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: NDArray, b: NDArray) -> NDArray:
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c


def get_warp_matrix(
    center: NDArray,
    scale: NDArray,
    rot: float,
    output_size: tuple[int, int],
    shift: tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
) -> NDArray:
    shift_arr = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift_arr
    src[1, :] = center + src_dir + scale * shift_arr
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat


def bbox_xyxy2cs(
    bbox: NDArray,
    padding: float = 1.25,
) -> tuple[NDArray, NDArray]:
    """Convert xyxy bbox → (center, scale) for top-down affine warp."""
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def top_down_affine(
    input_size: tuple[int, int],
    bbox_scale: NDArray,
    bbox_center: NDArray,
    img: NDArray,
) -> tuple[NDArray, NDArray]:
    """Affine-crop a person bbox from *img*, resize to *input_size*."""
    w, h = input_size
    warp_size = (int(w), int(h))

    aspect_ratio = w / h
    bw, bh = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(
        bw > bh * aspect_ratio,
        np.hstack([bw, bw / aspect_ratio]),
        np.hstack([bh * aspect_ratio, bh]),
    )

    warp_mat = get_warp_matrix(bbox_center, bbox_scale, 0.0, output_size=(w, h))
    img_out = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img_out, bbox_scale


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
# RTMPose top-down preprocessing / postprocessing
# ==========================================================================


def rtmpose_letterbox_preprocess(
    img: NDArray[np.uint8],
    bbox: NDArray[np.floating],
    model_input_size: tuple[int, int],
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """RTMPose-style top-down preprocessing: affine crop around bbox.

    Parameters
    ----------
    img : H×W×3 BGR uint8.
    bbox : [x1, y1, x2, y2] in image coordinates.
    model_input_size : (H, W) target size.
    mean, std : optional BGR normalisation.

    Returns
    -------
    resized_img : (H, W, 3) float32, normalised.
    center : (2,) float64  bbox center.
    scale : (2,) float64   bbox scale after aspect-ratio correction.
    """
    bbox_arr = np.array(bbox)
    center, scale = bbox_xyxy2cs(bbox_arr, padding=1.25)
    resized_img, scale = top_down_affine(model_input_size, scale, center, img)

    if mean is not None and std is not None:
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr = np.array(std, dtype=np.float32)
        resized_img = (resized_img.astype(np.float32) - mean_arr) / std_arr
    else:
        resized_img = resized_img.astype(np.float32)

    return resized_img, center, scale


def rtmpose_letterbox_postprocess(
    simcc_x: NDArray,
    simcc_y: NDArray,
    center: NDArray,
    scale: NDArray,
    model_input_size: tuple[int, int],
    simcc_split_ratio: float = 2.0,
) -> tuple[NDArray, NDArray]:
    """Decode SIMCC outputs back to original image coordinates.

    Parameters
    ----------
    simcc_x, simcc_y : (1, K, W) / (1, K, H) SIMCC heatmaps.
    center : (2,) bbox center in image coords.
    scale : (2,) bbox scale.
    model_input_size : (H, W) model input.
    simcc_split_ratio : label resolution divisor.

    Returns
    -------
    keypoints : (1, K, 2) float64  image-coordinate keypoints.
    scores : (1, K) float32
    """
    locs, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints = locs / simcc_split_ratio
    keypoints = keypoints / np.asarray(model_input_size) * scale
    keypoints = keypoints + center - scale / 2
    return keypoints.astype(np.float64), scores.astype(np.float32)
