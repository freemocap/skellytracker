"""
Per-camera person tracking state for YOLOX-skip logic.

When ``enable_tracking_skip`` is True, the centralized skeleton inference node
maintains one ``PersonTrackingState`` per camera. On frames where the tracking
lock is strong (high pose confidence, YOLOX re-run recently), YOLOX person
detection is skipped and the bounding box is predicted from a constant-velocity
motion model. The RTMPose pose estimator runs directly on the predicted crop.

This mirrors MediaPipe's tracking-confidence pattern, adapted for wall-clock
time: YOLOX is re-run at most once per second (configurable), plus immediately
whenever pose confidence drops below threshold or the person leaves the frame.

All functions in this module are stateless free functions that operate on the
``PersonTrackingState`` dataclass. The caller owns the states and threads them
through calls.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Per-skip expansion increment (pixels). Not user-facing — if you set this
# to zero, the predicted bbox never grows and tracking can drift out of the
# crop permanently without ever triggering a confidence drop.
# ---------------------------------------------------------------------------
_TRACKING_EXPANSION_PER_SKIP: float = 2.0


# ---------------------------------------------------------------------------
# Tracking state
# ---------------------------------------------------------------------------


@dataclass
class PersonTrackingState:
    """Per-camera tracking state for YOLOX-skip logic.

    Maintained externally to ``RTMPoseSession`` so the session stays
    stateless between calls. The caller passes these in and receives
    updated copies back.
    """

    # Current bounding box in xyxy format (image pixel coords).
    # None means no detection yet (cold start or lost track).
    bbox: NDArray[np.float64] | None = None

    # Bounding box center (cx, cy) in image pixel coords.
    center: NDArray[np.float64] | None = None

    # Bounding box size (width, height) in image pixel coords.
    size: NDArray[np.float64] | None = None

    # Smoothed velocity (dx, dy) in pixels per frame. EMA-updated from
    # frame-to-frame center displacement.
    velocity: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )

    # Average keypoint confidence from the most recent RTMPose result.
    # Used as the quality signal for the tracking-vs-redetect decision.
    pose_confidence: float = 0.0

    # Number of consecutive frames where we skipped YOLOX (tracking only).
    consecutive_skips: int = 0

    # ``time.perf_counter()`` of the last full YOLOX detection.
    # 0.0 = never (cold start). Updated by ``update_tracking_state`` when
    # ``from_detector=True``.
    last_detection_time: float = 0.0

    @property
    def is_valid(self) -> bool:
        """True if we have a tracking lock (at least one successful detection)."""
        return self.bbox is not None and self.center is not None


# ---------------------------------------------------------------------------
# Motion model: predict where the person will be next frame
# ---------------------------------------------------------------------------


def predict_bbox_from_tracking(
    state: PersonTrackingState,
    *,
    velocity_alpha: float = 0.7,
    expansion_ratio: float = 0.1,
    expansion_per_skip: float = _TRACKING_EXPANSION_PER_SKIP,
    image_width: int,
    image_height: int,
) -> NDArray[np.float64] | None:
    """Predict the person bounding box for the next frame.

    Uses a constant-velocity model: predicted center = previous center +
    smoothed velocity. The bbox is expanded on all sides to account for
    prediction uncertainty, with the expansion growing linearly with
    ``consecutive_skips`` to prevent drift escape.

    Returns None if the state is not valid or the predicted bbox collapses
    to zero area (person walked out of frame).
    """
    if not state.is_valid:
        return None

    # Predict center using constant-velocity model.
    predicted_center = state.center + state.velocity

    # Expand bbox to account for prediction uncertainty.
    # Progressive expansion with skip count catches slow drift.
    expansion = (
        expansion_ratio * max(state.size[0], state.size[1])
        + expansion_per_skip * state.consecutive_skips
    )
    predicted_half_w = state.size[0] / 2.0 + expansion
    predicted_half_h = state.size[1] / 2.0 + expansion

    x1 = predicted_center[0] - predicted_half_w
    y1 = predicted_center[1] - predicted_half_h
    x2 = predicted_center[0] + predicted_half_w
    y2 = predicted_center[1] + predicted_half_h

    # Clamp to image bounds.
    x1 = max(0.0, float(x1))
    y1 = max(0.0, float(y1))
    x2 = min(float(image_width), float(x2))
    y2 = min(float(image_height), float(y2))

    # If the predicted bbox collapsed to nothing, return None (forces re-detect).
    if x2 <= x1 or y2 <= y1:
        return None

    return np.array([x1, y1, x2, y2], dtype=np.float64)


# ---------------------------------------------------------------------------
# State update: consume new detection results
# ---------------------------------------------------------------------------


def update_tracking_state(
    state: PersonTrackingState,
    bbox: NDArray[np.float64] | None,
    scores: NDArray[np.float32] | None,
    *,
    velocity_alpha: float = 0.7,
    from_detector: bool = False,
) -> PersonTrackingState:
    """Update tracking state from a new detection (YOLOX or tracking-based).

    Args:
        state: Previous tracking state.
        bbox: The new bbox in xyxy format. For multi-person results the
              first row is used. None or empty = detection lost this frame.
        scores: RTMPose keypoint scores ``(K,)`` or ``(N, K)``. Used to
                compute the mean pose confidence. None = no scores available.
        velocity_alpha: EMA smoothing factor for velocity update.
        from_detector: True when the bbox came from a full YOLOX run
                       (updates ``last_detection_time``). False when the
                       bbox was tracking-predicted.

    Returns:
        A new ``PersonTrackingState`` reflecting the latest detection.
        Returns a fresh cold-start state when detection is lost.
    """
    if bbox is None or len(bbox) == 0:
        # Detection lost this frame. Return fresh cold-start state so the
        # next frame forces a full YOLOX re-detection.
        return PersonTrackingState()

    # Take first person bbox (multi-person identity tracking is out of scope).
    bbox_arr = np.asarray(
        bbox[0] if bbox.ndim == 2 else bbox,
        dtype=np.float64,
    )

    new_center = np.array(
        [
            (bbox_arr[0] + bbox_arr[2]) / 2.0,
            (bbox_arr[1] + bbox_arr[3]) / 2.0,
        ],
        dtype=np.float64,
    )
    new_size = np.array(
        [
            bbox_arr[2] - bbox_arr[0],
            bbox_arr[3] - bbox_arr[1],
        ],
        dtype=np.float64,
    )

    # Compute new velocity from displacement (EMA-smoothed).
    if state.is_valid:
        raw_velocity = new_center - state.center
        new_velocity = (
            velocity_alpha * state.velocity
            + (1.0 - velocity_alpha) * raw_velocity
        )
    else:
        new_velocity = state.velocity.copy()  # keep zeros on cold start

    # Compute average keypoint confidence.
    if scores is not None and len(scores) > 0:
        scores_arr = np.asarray(scores, dtype=np.float32)
        if scores_arr.ndim == 2:
            scores_arr = scores_arr[0]  # first person
        pose_conf = float(np.mean(scores_arr))
    else:
        pose_conf = 0.0

    return PersonTrackingState(
        bbox=bbox_arr,
        center=new_center,
        size=new_size,
        velocity=new_velocity,
        pose_confidence=pose_conf,
        consecutive_skips=0,  # reset on every successful result
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
    predicted_bbox: NDArray[np.float64] | None,
) -> bool:
    """Decide whether to run full YOLOX detection for this camera.

    Returns True if YOLOX should run (tracking is stale / lost / unconfident).
    Returns False if the tracking-predicted bbox is good enough.

    The decision is a logical OR of several safety conditions — any single
    one can force a re-detection.
    """
    # Cold start (last_detection_time == 0.0 means never detected): always run.
    if not state.is_valid or state.last_detection_time == 0.0:
        return True

    # Tracking confidence too low: re-detect.
    if state.pose_confidence < min_tracking_confidence:
        return True

    # Haven't run YOLOX in the last ``min_detection_interval`` seconds:
    # time for a periodic refresh.
    if time.perf_counter() - state.last_detection_time >= min_detection_interval:
        return True

    # Predicted bbox is None (drifted out of frame): re-detect.
    if predicted_bbox is None:
        return True

    # Predicted bbox collapsed to <25% of expected area: re-detect.
    pred_w = predicted_bbox[2] - predicted_bbox[0]
    pred_h = predicted_bbox[3] - predicted_bbox[1]
    if state.size is not None:
        expected_area = float(state.size[0] * state.size[1])
        pred_area = float(pred_w * pred_h)
        if expected_area > 0 and pred_area < 0.25 * expected_area:
            return True

    return False
