"""
Per-camera person tracking state for YOLOX-skip logic.

On frames where tracking is active, YOLOX is skipped and the bounding box
is predicted by expanding the previous frame's keypoint-derived bbox by a
configurable margin. The RTMPose pose estimator runs directly on this
expanded crop. YOLOX re-runs periodically (~once per second) and whenever
pose confidence drops.

This is simpler and more robust than a velocity-based motion model: the
bbox follows the person naturally because it's recomputed from the actual
keypoint positions each frame.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class PersonTrackingState:
    """Per-camera tracking state for YOLOX-skip logic."""

    # Current bounding box in xyxy format (image pixel coords).
    # None = cold start or lost track.
    bbox: NDArray | None = None

    # Average keypoint confidence from the most recent RTMPose result.
    pose_confidence: float = 0.0

    # Consecutive frames since last YOLOX re-detection.
    consecutive_skips: int = 0

    # ``time.perf_counter()`` of the last full YOLOX detection.
    # 0.0 = never (cold start).
    last_detection_time: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.bbox is not None


# ---------------------------------------------------------------------------
# Predict bbox for next frame: expand the current bbox by a margin
# ---------------------------------------------------------------------------


def predict_bbox_from_tracking(
    state: PersonTrackingState,
    *,
    expansion_ratio: float = 0.05,
    image_width: int,
    image_height: int,
) -> NDArray | None:
    """Expand the current bbox by ``expansion_ratio`` on all sides.

    Returns None if the state is invalid or the expanded bbox collapses.
    """
    if not state.is_valid:
        return None

    x1, y1, x2, y2 = state.bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    expand_w = w * expansion_ratio
    expand_h = h * expansion_ratio

    x1 = max(0.0, float(x1 - expand_w))
    y1 = max(0.0, float(y1 - expand_h))
    x2 = min(float(image_width), float(x2 + expand_w))
    y2 = min(float(image_height), float(y2 + expand_h))

    if x2 <= x1 or y2 <= y1:
        return None

    return np.array([x1, y1, x2, y2], dtype=np.float64)


# ---------------------------------------------------------------------------
# Update state from RTMPose results: compute tight keypoint bbox, expand
# ---------------------------------------------------------------------------


def _keypoints_to_bbox(
    keypoints: NDArray,
    scores: NDArray,
    conf_threshold: float = 0.3,
) -> NDArray | None:
    """Compute a tight xyxy bbox around visible keypoints.

    Args:
        keypoints: (K, 2) or (N, K, 2).
        scores: (K,) or (N, K).
    Returns (4,) xyxy or None if no visible keypoints.
    """
    if keypoints.ndim == 2:
        keypoints = keypoints[None, ...]
        scores = scores[None, ...]

    all_pts = []
    for inst in range(keypoints.shape[0]):
        kp = keypoints[inst]
        sc = scores[inst]
        for i in range(len(sc)):
            if sc[i] >= conf_threshold:
                x, y = kp[i]
                if np.isfinite(x) and np.isfinite(y):
                    all_pts.append((x, y))

    if not all_pts:
        return None

    pts = np.array(all_pts, dtype=np.float64)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)

    if x2 <= x1 or y2 <= y1:
        return None

    return np.array([x1, y1, x2, y2], dtype=np.float64)


def update_tracking_state(
    state: PersonTrackingState,
    keypoints: NDArray | None,
    scores: NDArray | None,
    *,
    expansion_ratio: float = 0.05,
    from_detector: bool = False,
    image_width: int = 99999,
    image_height: int = 99999,
    kpt_conf_threshold: float = 0.3,
) -> PersonTrackingState:
    """Update tracking state from a new RTMPose result.

    Computes a tight bbox around visible keypoints, expands it by
    ``expansion_ratio``, and stores it as the prediction for the next frame.

    Args:
        state: Previous state.
        keypoints: RTMPose keypoints (K, 2) or (N, K, 2). None = no detection.
        scores: RTMPose scores (K,) or (N, K). None = no detection.
        expansion_ratio: How much to expand the tight keypoint bbox.
        from_detector: True if YOLOX ran (updates last_detection_time).
        image_width, image_height: Image dimensions for clamping.
        kpt_conf_threshold: Min score to consider a keypoint visible.
    """
    if keypoints is None or scores is None or len(keypoints) == 0 or len(scores) == 0:
        return PersonTrackingState()

    kp_bbox = _keypoints_to_bbox(keypoints, scores, kpt_conf_threshold)
    if kp_bbox is None:
        return PersonTrackingState()

    # Expand and clamp.
    x1, y1, x2, y2 = kp_bbox
    w = x2 - x1
    h = y2 - y1
    expand_w = w * expansion_ratio
    expand_h = h * expansion_ratio

    x1 = max(0.0, float(x1 - expand_w))
    y1 = max(0.0, float(y1 - expand_h))
    x2 = min(float(image_width), float(x2 + expand_w))
    y2 = min(float(image_height), float(y2 + expand_h))
    expanded_bbox = np.array([x1, y1, x2, y2], dtype=np.float64)

    # Confidence.
    scores_arr = np.asarray(scores, dtype=np.float32)
    if scores_arr.ndim == 2:
        scores_arr = scores_arr[0]
    pose_conf = float(np.mean(scores_arr)) if len(scores_arr) > 0 else 0.0

    return PersonTrackingState(
        bbox=expanded_bbox,
        pose_confidence=pose_conf,
        consecutive_skips=0,
        last_detection_time=(
            time.perf_counter() if from_detector else state.last_detection_time
        ),
    )


# ---------------------------------------------------------------------------
# Re-detection gate
# ---------------------------------------------------------------------------


def should_run_detector(
    state: PersonTrackingState,
    *,
    min_tracking_confidence: float = 0.3,
    min_detection_interval: float = 1.0,
    predicted_bbox: NDArray | None,
) -> bool:
    """Decide whether to run full YOLOX detection for this camera."""
    if not state.is_valid or state.last_detection_time == 0.0:
        return True
    if state.pose_confidence < min_tracking_confidence:
        return True
    if time.perf_counter() - state.last_detection_time >= min_detection_interval:
        return True
    if predicted_bbox is None:
        return True
    pred_w = predicted_bbox[2] - predicted_bbox[0]
    pred_h = predicted_bbox[3] - predicted_bbox[1]
    if state.bbox is not None:
        expected_w = state.bbox[2] - state.bbox[0]
        expected_h = state.bbox[3] - state.bbox[1]
        expected_area = float(expected_w * expected_h)
        pred_area = float(pred_w * pred_h)
        if expected_area > 0 and pred_area < 0.25 * expected_area:
            return True
    return False
