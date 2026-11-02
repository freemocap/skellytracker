"""Palm alignment for the experimental ONNX hand model.

Algorithm conventions follow the pinned OpenCV Zoo adapter [@opencvZooHand2023].
Outer hand crops remain owned by mediapipe_hand_stage. These transforms are the
model's internal palm alignment, not a replacement hand-crop policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from skellytracker.core.detectors.metadata import ImageTransform
from skellytracker.core.detectors.processing.image_preprocessing import preprocess_image
from skellytracker.core.sidecar.model import InputSpec

PALM_KEYPOINT_NAMES = (
    "wrist",
    "index_mcp",
    "middle_mcp",
    "ring_mcp",
    "pinky_mcp",
    "thumb_cmc",
    "thumb_mcp",
)
PALM_ORIGIN_INDEX = PALM_KEYPOINT_NAMES.index("wrist")
PALM_DIRECTION_INDEX = PALM_KEYPOINT_NAMES.index("middle_mcp")


@dataclass(frozen=True, slots=True)
class PalmDetection:
    box_corners: NDArray[np.float64]
    keypoint_xy: NDArray[np.float64]
    confidence: float


@dataclass(frozen=True, slots=True)
class HandAlignment:
    model_to_crop: NDArray[np.float64]
    depth_scale: float

    def restore_keypoints(self, model_xyz: NDArray) -> NDArray[np.float64]:
        values = np.array(model_xyz, dtype=np.float64, copy=True)
        if values.ndim != 2 or values.shape[-1] != 3:
            raise ValueError("Hand image coordinates require xyz components")
        values[:, :2] = (
            values[:, :2] @ self.model_to_crop[:2, :2].T + self.model_to_crop[:2, 2]
        )
        values[:, 2] *= self.depth_scale
        return values


@lru_cache(maxsize=1)
def palm_anchor_centers(input_size: tuple[int, int]) -> NDArray[np.float64]:
    """Generate the pinned palm model's two SSD grids once, not on each frame."""
    grids = []
    for stride, repeat_count in ((8, 2), (16, 6)):
        height, width = input_size
        if height % stride or width % stride:
            raise ValueError("Palm input dimensions must match SSD strides")
        rows, columns = np.meshgrid(
            (np.arange(height // stride) + 0.5) / (height // stride),
            (np.arange(width // stride) + 0.5) / (width // stride),
            indexing="ij",
        )
        centers = np.stack((columns, rows), axis=-1).reshape(-1, 2)
        grids.append(np.repeat(centers, repeat_count, axis=0))
    result = np.concatenate(grids)
    result.flags.writeable = False
    return result


def decode_single_palm(
    offsets: NDArray,
    logits: NDArray,
    input_size: tuple[int, int],
    image_transform: ImageTransform,
    confidence_threshold: float,
) -> PalmDetection | None:
    anchors = palm_anchor_centers(input_size)
    expected_width = 4 + len(PALM_KEYPOINT_NAMES) * 2
    if offsets.shape != (len(anchors), expected_width) or logits.shape != (
        len(anchors),
        1,
    ):
        raise ValueError("Palm output signature does not match the pinned model")
    scores = expit(logits[:, 0])
    valid = np.isfinite(scores) & np.isfinite(offsets).all(axis=-1)
    if not valid.any():
        return None
    selected = int(np.argmax(np.where(valid, scores, -np.inf)))
    if scores[selected] < confidence_threshold:
        return None
    values = offsets[selected].astype(np.float64)
    height, width = input_size
    center = values[:2] + anchors[selected] * (width, height)
    half_size = values[2:4] / 2
    if np.any(half_size <= 0):
        return None
    corners = image_transform.restore_points(
        np.stack((center - half_size, center + half_size))
    )
    points = image_transform.restore_points(
        values[4:].reshape(-1, 2) + anchors[selected] * (width, height)
    )
    return PalmDetection(corners, points, float(scores[selected]))


def _crop_and_pad(image: NDArray, corners: NDArray, *, rotation_margin: bool):
    # Pinned model alignment parameters, separate from the shared outer 3x crop.
    extent = corners[1] - corners[0]
    center = corners.mean(axis=0) + extent * ((0, 0) if rotation_margin else (0, -0.4))
    half_size = extent * (4.0 if rotation_margin else 3.0) / 2
    clipped = np.stack((center - half_size, center + half_size)).astype(np.int64)
    np.clip(clipped, (0, 0), (image.shape[1], image.shape[0]), out=clipped)
    crop = image[clipped[0, 1] : clipped[1, 1], clipped[0, 0] : clipped[1, 0]]
    if not crop.size:
        return None
    side = (
        int(np.linalg.norm(crop.shape[:2])) if rotation_margin else max(crop.shape[:2])
    )
    padding = np.array((side - crop.shape[1], side - crop.shape[0]))
    before = padding // 2
    after = padding - before
    padded = cv2.copyMakeBorder(
        crop,
        int(before[1]),
        int(after[1]),
        int(before[0]),
        int(after[0]),
        cv2.BORDER_CONSTANT,
    )
    return padded, clipped, clipped[0] - before


def prepare_aligned_hand(
    image: NDArray,
    palm: PalmDetection,
    input_spec: InputSpec,
    input_size: tuple[int, int],
    *,
    normalization_fn: Callable[[NDArray[np.uint8]], NDArray[np.float32]] | None = None,
) -> tuple[NDArray[np.float32], HandAlignment] | None:
    if not image.size or not np.isfinite(palm.keypoint_xy).all():
        return None
    first = _crop_and_pad(image, palm.box_corners, rotation_margin=True)
    if first is None:
        return None
    padded, corners, first_bias = first
    points = palm.keypoint_xy - first_bias
    direction = points[PALM_DIRECTION_INDEX] - points[PALM_ORIGIN_INDEX]
    if np.linalg.norm(direction) < np.finfo(np.float64).eps:
        return None
    radians = np.pi / 2 - np.arctan2(-direction[1], direction[0])
    radians -= 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
    rotation = cv2.getRotationMatrix2D(
        tuple((corners - first_bias).mean(axis=0)), np.degrees(radians), 1
    )
    rotated = cv2.warpAffine(padded, rotation, (padded.shape[1], padded.shape[0]))
    rotated_points = points @ rotation[:, :2].T + rotation[:, 2]
    second = _crop_and_pad(
        rotated,
        np.stack((rotated_points.min(axis=0), rotated_points.max(axis=0))),
        rotation_margin=False,
    )
    if second is None:
        return None
    aligned_image, _, second_bias = second
    tensor, resize_transform = preprocess_image(
        aligned_image,
        input_size,
        input_spec=input_spec,
        precision="fp32",
        normalization_fn=normalization_fn,
    )
    if not isinstance(resize_transform, ImageTransform):
        raise ValueError("Hand alignment requires an explicit resize transform")
    inverse_rotation = np.eye(3)
    inverse_rotation[:2] = cv2.invertAffineTransform(rotation)
    restore_resize = np.eye(3)
    restore_resize[0, 0] = 1 / resize_transform.scale_x
    restore_resize[1, 1] = 1 / resize_transform.scale_y
    restore_resize[:2, 2] = second_bias - np.array(
        (
            resize_transform.offset_x / resize_transform.scale_x,
            resize_transform.offset_y / resize_transform.scale_y,
        )
    )
    model_to_crop = inverse_rotation @ restore_resize
    model_to_crop[:2, 2] += first_bias
    return tensor, HandAlignment(model_to_crop, 1 / resize_transform.scale_x)
