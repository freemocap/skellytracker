"""
Stateless ROI crop utilities extracted from MediapipeCompositeDetector.

These are the core geometry functions for computing square ROI crops from
body keypoint positions. They are tracker-agnostic — they work with any
body keypoint source (RTMO, RTMPose, MediaPipe, etc.) as long as the
caller provides (x, y) coordinates in image pixel space.

State management (last hand sizes, smoothed ROI centers/sizes, handedness
tracking) belongs in the caller. These functions are pure geometry.
"""

import numpy as np


class ROIBox:
    """Bounding box for an ROI crop in full-image coordinates."""

    __slots__ = ("x", "y", "width", "height")

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """Slice the crop region from an image. Returns a copy."""
        return image[self.y : self.y + self.height, self.x : self.x + self.width].copy()


def compute_square_roi(
    center_x: int,
    center_y: int,
    size: int,
    image_w: int,
    image_h: int,
) -> ROIBox:
    """Compute a square ROI clamped to image bounds."""
    half = size // 2
    x = max(0, center_x - half)
    y = max(0, center_y - half)
    x2 = min(image_w, center_x + half)
    y2 = min(image_h, center_y + half)
    return ROIBox(x=x, y=y, width=x2 - x, height=y2 - y)


def smooth_roi_params(
    *,
    raw_cx: float,
    raw_cy: float,
    raw_size: float,
    prev_smoothed: tuple[float, float, float] | None,
    alpha: float = 0.5,
) -> tuple[float, float, float]:
    """
    Apply exponential moving average smoothing to ROI center and size.

    Args:
        raw_cx: Raw center X from this frame's landmarks.
        raw_cy: Raw center Y from this frame's landmarks.
        raw_size: Raw crop size from this frame's computation.
        prev_smoothed: Previous (cx, cy, size) tuple, or None on cold start.
        alpha: Smoothing factor. 0 = no smoothing (use raw), 1 = frozen.

    Returns:
        Smoothed (center_x, center_y, size).
    """
    if prev_smoothed is None:
        return (raw_cx, raw_cy, raw_size)

    prev_cx, prev_cy, prev_size = prev_smoothed
    return (
        alpha * prev_cx + (1.0 - alpha) * raw_cx,
        alpha * prev_cy + (1.0 - alpha) * raw_cy,
        alpha * prev_size + (1.0 - alpha) * raw_size,
    )


def hand_bbox_diagonal(landmarks_xyz: np.ndarray) -> float:
    """Compute the bounding box diagonal of hand landmarks in pixel space."""
    points_2d = landmarks_xyz[:, :2]
    valid = points_2d[~np.isnan(points_2d).any(axis=1)]
    if len(valid) < 2:
        return 0.0
    mins = valid.min(axis=0)
    maxs = valid.max(axis=0)
    return float(np.linalg.norm(maxs - mins))


def compute_hand_crop_size(
    *,
    arm_length: float,
    last_hand_diagonal: float = 0.0,
    image_h: int,
    hand_roi_scale: float = 2.0,
    hand_bbox_padding: float = 1.8,
    min_hand_crop_image_fraction: float = 0.15,
) -> float:
    """
    Compute the hand crop size from multiple independent measures.

    Each measure uses its own multiplier — they are not stacked.
    The maximum of the three is used for robustness.

    Args:
        arm_length: Distance from wrist to elbow in pixels.
        last_hand_diagonal: Previous frame's hand bbox diagonal (0 on cold start).
        image_h: Image height in pixels.
        hand_roi_scale: Multiplier on arm length (default 2.0).
        hand_bbox_padding: Multiplier on previous hand bbox diagonal (default 1.8).
        min_hand_crop_image_fraction: Minimum crop as fraction of image height (default 0.15).

    Returns:
        Crop size in pixels.
    """
    arm_crop = arm_length * hand_roi_scale
    hand_crop = last_hand_diagonal * hand_bbox_padding if last_hand_diagonal > 0.0 else 0.0
    min_crop = float(image_h) * min_hand_crop_image_fraction
    return max(arm_crop, hand_crop, min_crop)


def collect_visible_head_points(
    *,
    body_xyz: np.ndarray,
    body_vis: np.ndarray,
    head_indices: list[int],
    visibility_threshold: float = 0.5,
) -> np.ndarray | None:
    """
    Collect 2D positions of head landmarks exceeding the visibility threshold.

    Args:
        body_xyz: (N, 3) array of body keypoints in pixel coords.
        body_vis: (N,) array of visibility scores.
        head_indices: Which body keypoint indices correspond to head anatomy.
        visibility_threshold: Minimum visibility to include a point.

    Returns:
        (M, 2) array of visible head points, or None if fewer than 2 are visible.
    """
    points: list[np.ndarray] = []
    for idx in head_indices:
        if body_vis[idx] >= visibility_threshold:
            xy = body_xyz[idx, :2]
            if not np.isnan(xy).any():
                points.append(xy)

    if len(points) < 2:
        return None
    return np.stack(points, axis=0)


def compute_face_crop_params(
    *,
    visible_head_points: np.ndarray,
    face_roi_scale: float = 2.5,
) -> tuple[tuple[float, float], float] | None:
    """
    Compute the face ROI center and crop size from visible head landmarks.

    Args:
        visible_head_points: (M, 2) array of visible head keypoints.
        face_roi_scale: Multiplier on the larger head bbox dimension.

    Returns:
        ((center_x, center_y), crop_size) or None if input has < 2 points.
    """
    if visible_head_points.shape[0] < 2:
        return None

    min_xy = visible_head_points.min(axis=0)
    max_xy = visible_head_points.max(axis=0)
    bbox_center = (min_xy + max_xy) / 2.0
    bbox_w = float(max_xy[0] - min_xy[0])
    bbox_h = float(max_xy[1] - min_xy[1])

    crop_size = max(bbox_w, bbox_h) * face_roi_scale
    if crop_size <= 1.0:
        return None

    return (float(bbox_center[0]), float(bbox_center[1])), crop_size
