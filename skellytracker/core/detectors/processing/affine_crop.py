"""Generic top-down affine-crop preprocessing.

Vendored from rtmlib (``tools/pose_estimation/pre_processings.py``). Despite
the rtmlib origin, none of this math is specific to RTMPose — it's the
standard bbox-to-affine-crop transform used across top-down 2D pose
estimators generally: any detector that crops a person/region bbox, warps it
to a fixed input size, and needs to unproject predictions back to image
space builds on this — both `image_preprocessing.py`'s `affine_person_crop`
resize method and `simcc_decode.py`'s postprocessing do.

Free functions only — no detector-family dependencies.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


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
