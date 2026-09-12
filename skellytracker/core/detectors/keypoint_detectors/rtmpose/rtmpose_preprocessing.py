"""RTMPose-specific preprocessing and postprocessing.

Vendored from rtmlib (``tools/pose_estimation/rtmpose.py`` and
``tools/pose_estimation/pre_processings.py``).  All functions are free
functions — no rtmlib class dependencies.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


# ==========================================================================
# SIMCC decoding
# ==========================================================================


def get_simcc_maximum(
    simcc_x: NDArray,
    simcc_y: NDArray,
) -> tuple[NDArray, NDArray]:
    """Decode SIMCC heatmaps to (x, y) coordinates with confidence scores.

    Confidence is the *raw* SIMCC peak response averaged across the two axes,
    matching mmpose and rtmlib.  RTMPose is trained with a Gaussian-target KL
    loss, so a well-localised keypoint drives its winning bin to ~1.0 while an
    occluded one leaves a low, flat response — the peak height itself is the
    confidence signal.

    Do NOT softmax the bins first.  Softmax normalises each keypoint's response
    to sum to 1 over its bins, which discards peak height entirely: every
    keypoint then scores within a hair of the uniform floor (~1/Wx, ~1/Wy) and
    confident points stop separating from occluded ones.  Coordinates survive
    that mistake because argmax is monotone under softmax, so it goes unnoticed
    unless the confidences are inspected directly.

    Values are unbounded above (peaks above 1.0 are routine); the [0, 1] clip
    that ``Keypoints.visibility`` requires happens in
    ``rtmpose_letterbox_postprocess``, which is where skellytracker's contract
    starts.  This function stays byte-comparable with upstream mmpose/rtmlib.

    Parameters
    ----------
    simcc_x : np.ndarray  shape (N, K, Wx) — SIMCC responses for the x-axis.
    simcc_y : np.ndarray  shape (N, K, Wy) — SIMCC responses for the y-axis.

    Returns
    -------
    locs : np.ndarray  shape (N, K, 2)  x/y keypoint coordinates (bin indices).
    vals : np.ndarray  shape (N, K)     confidence = mean of the x and y peak
                                        responses.  Unbounded above; <= 0.0
                                        marks a non-detection (locs set to -1).
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)

    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    vals = 0.5 * (max_val_x + max_val_y)
    locs[vals <= 0.0] = -1

    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


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
    scores : (1, K) float32  confidence in [0, 1].

    Notes
    -----
    Raw SIMCC peaks are unbounded above, so they are clipped into [0, 1] here to
    satisfy ``Keypoints.visibility``.  That contract is shared with the MediaPipe
    detectors, so one threshold has to mean the same thing across backends.  The
    clip costs nothing: it only touches peaks already above 1.0, which clear
    every threshold <= 1.0 either way, and no consumer uses visibility as a
    weight — only as a threshold.
    """
    locs, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints = locs / simcc_split_ratio
    keypoints = keypoints / np.asarray(model_input_size) * scale
    keypoints = keypoints + center - scale / 2
    scores = np.clip(scores, 0.0, 1.0)
    return keypoints.astype(np.float64), scores.astype(np.float32)
