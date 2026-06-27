"""
Per-camera person tracking state for YOLOX-skip logic.

On frames where tracking is active, YOLOX is skipped and the bounding box
is predicted by expanding the previous frame's keypoint-derived bbox by a
configurable margin. The RTMPose pose estimator runs directly on this
expanded crop. YOLOX re-runs periodically (~every few seconds) and whenever
pose confidence drops.

This is simpler and more robust than a velocity-based motion model: the
bbox follows the person naturally because it's recomputed from the actual
keypoint positions each frame.

..  note::

    **Confidence semantics** — ``pose_confidence`` is a float in [0, 1] where
    higher = better track quality.  For RTMPose / SIMCC it is the *fraction
    of keypoints whose softmax peak exceeds the visibility threshold*.

    This is tracker-agnostic by design: any tracker that produces per-keypoint
    scores can plug into ``update_tracking_state`` by providing a
    ``kpt_visibility_threshold`` appropriate for its score distribution.

    **SIMCC score background** — SIMCC applies softmax over *N* spatial bins
    (typically 64–128).  The uniform-distribution baseline is ≈ 1/N (0.008−0.016).
    A sharp peak may reach 0.05−0.20.  The old code used 0.3 as both the
    bbox-inclusion threshold and the tracking-confidence threshold, which is
    *above the maximum possible SIMCC peak* — zero keypoints passed, the
    tracking state was reset every frame, and YOLOX ran on every frame.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Debug: log SIMCC score distribution once every N frames so we can see the
# actual range and tune thresholds accordingly.
# ---------------------------------------------------------------------------
_DEBUG_SCORE_LOG_INTERVAL = 30  # frames
_debug_score_frame_counter: int = 0


def _debug_log_score_distribution(
    scores: NDArray,
    bbox_threshold: float,
    visibility_threshold: float,
) -> None:
    global _debug_score_frame_counter
    _debug_score_frame_counter += 1
    if _debug_score_frame_counter % _DEBUG_SCORE_LOG_INTERVAL != 1:
        return

    arr = np.asarray(scores, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if len(arr) == 0:
        return

    pct = [0, 10, 25, 50, 75, 90, 100]
    vals = np.percentile(arr, pct)
    n_bbox = int(np.sum(arr >= bbox_threshold))
    n_vis = int(np.sum(arr >= visibility_threshold))
    parts = "  ".join(f"p{p}={vals[i]:.4f}" for i, p in enumerate(pct))

# ---------------------------------------------------------------------------
# Default thresholds — tuned for RTMPose SIMCC softmax-peak scores.
#
# SIMCC applies softmax over hundreds of spatial bins, so even the sharpest
# keypoint peak is tiny (0.002–0.007).  These thresholds were calibrated
# against real score distributions from the rtmw-dw-l-m model.
#
# If you add a new tracker type whose per-keypoint scores live in a
# different numeric range, these are the two knobs to adjust (or expose
# per-tracker in the config).
# ---------------------------------------------------------------------------

# Minimum SIMCC softmax peak to include a keypoint in the tight-person-bbox
# computation.  0.003 sits just above the noise floor (~0.0025) so we
# include every keypoint with any hint of signal.
_DEFAULT_KPT_BBOX_THRESHOLD: float = 0.003

# Minimum SIMCC softmax peak to count a keypoint as "visible" for the
# pose-confidence fraction.  0.004 captures the top ~50% of keypoints
# on a typical frame and drops sharply when the person is occluded.
_DEFAULT_KPT_VISIBILITY_THRESHOLD: float = 0.004


@dataclass
class PersonTrackingState:
    """Per-camera tracking state for YOLOX-skip logic.

    Attributes:
        bbox:
            Current bounding box in xyxy format (image pixel coords).
            ``None`` = cold start or lost track.
        pose_confidence:
            Track quality in [0, 1].  For RTMPose this is the fraction of
            keypoints whose SIMCC softmax peak exceeds the visibility
            threshold.  Higher = more of the skeleton is clearly visible.
        consecutive_skips:
            Consecutive frames since the last YOLOX re-detection.
        last_detection_time:
            ``time.perf_counter()`` of the last full YOLOX detection.
            ``0.0`` = never (cold start).
    """

    bbox: NDArray | None = None
    pose_confidence: float = 0.0
    consecutive_skips: int = 0
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

    Returns ``None`` if the state is invalid or the expanded bbox collapses.
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
# Update state from RTMPose results
# ---------------------------------------------------------------------------


def _keypoints_to_bbox(
    keypoints: NDArray,
    scores: NDArray,
    conf_threshold: float = _DEFAULT_KPT_BBOX_THRESHOLD,
) -> NDArray | None:
    """Compute a tight xyxy bbox around keypoints whose score ≥ *conf_threshold*.

    Args:
        keypoints: ``(K, 2)`` or ``(N, K, 2)``.
        scores: ``(K,)`` or ``(N, K)``.
        conf_threshold: Minimum per-keypoint score for inclusion.

    Returns:
        ``(4,)`` xyxy array or ``None`` if no keypoint passes the threshold.
    """
    if keypoints.ndim == 2:
        keypoints = keypoints[None, ...]
        scores = scores[None, ...]

    all_pts: list[tuple[float, float]] = []
    for inst in range(keypoints.shape[0]):
        kp = keypoints[inst]
        sc = scores[inst]
        for i in range(len(sc)):
            if sc[i] >= conf_threshold:
                x, y = kp[i]
                if np.isfinite(x) and np.isfinite(y):
                    all_pts.append((float(x), float(y)))

    if not all_pts:
        return None

    pts = np.array(all_pts, dtype=np.float64)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)

    if x2 <= x1 or y2 <= y1:
        return None

    return np.array([x1, y1, x2, y2], dtype=np.float64)


def _compute_pose_confidence(
    scores: NDArray,
    *,
    visibility_threshold: float = _DEFAULT_KPT_VISIBILITY_THRESHOLD,
) -> float:
    """Compute track-quality confidence as the fraction of keypoints whose
    score exceeds *visibility_threshold*.

    Result is in [0, 1]: 0 = no keypoints clearly visible, 1 = entire
    skeleton is sharp.  This is tracker-agnostic — any tracker that produces
    per-keypoint scores can use it, as long as *visibility_threshold* is
    tuned for that tracker's score distribution.
    """
    scores_arr = np.asarray(scores, dtype=np.float32)
    if scores_arr.ndim == 2:
        scores_arr = scores_arr[0]  # (N, K) → (K,) — take first person
    if len(scores_arr) == 0:
        return 0.0
    return float(np.mean(scores_arr >= visibility_threshold))


def update_tracking_state(
    state: PersonTrackingState,
    keypoints: NDArray | None,
    scores: NDArray | None,
    *,
    expansion_ratio: float = 0.05,
    from_detector: bool = False,
    image_width: int = 99999,
    image_height: int = 99999,
    kpt_bbox_threshold: float = _DEFAULT_KPT_BBOX_THRESHOLD,
    kpt_visibility_threshold: float = _DEFAULT_KPT_VISIBILITY_THRESHOLD,
) -> PersonTrackingState:
    """Update tracking state from a new RTMPose result.

    Computes a tight bbox around visible keypoints, expands it by
    *expansion_ratio*, and stores it as the prediction for the next frame.
    Confidence is the fraction of keypoints above *kpt_visibility_threshold*.

    Args:
        state: Previous state.
        keypoints: RTMPose keypoints ``(K, 2)`` or ``(N, K, 2)``.  ``None`` = no detection.
        scores: RTMPose scores ``(K,)`` or ``(N, K)``.  ``None`` = no detection.
        expansion_ratio: How much to expand the tight keypoint bbox.
        from_detector: ``True`` if YOLOX ran (updates ``last_detection_time``).
        image_width: Image width in pixels (for clamping).
        image_height: Image height in pixels (for clamping).
        kpt_bbox_threshold: Minimum score to include a keypoint in the bbox.
            For SIMCC this should be just above the uniform baseline (~0.02).
        kpt_visibility_threshold: Minimum score to count a keypoint as "visible"
            for the pose-confidence fraction.  For SIMCC ~0.05 is reasonable.
    """
    if keypoints is None or scores is None or len(keypoints) == 0 or len(scores) == 0:
        return PersonTrackingState()

    # --- Debug: log SIMCC score distribution once per second ---
    _debug_log_score_distribution(scores, kpt_bbox_threshold, kpt_visibility_threshold)

    kp_bbox = _keypoints_to_bbox(keypoints, scores, kpt_bbox_threshold)
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

    pose_conf = _compute_pose_confidence(scores, visibility_threshold=kpt_visibility_threshold)

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
    """Decide whether to run full YOLOX detection for this camera.

    Returns ``True`` (run YOLOX) when any of these fire:

    * Cold start (state invalid or never detected).
    * ``pose_confidence`` dropped below *min_tracking_confidence*.
    * *min_detection_interval* seconds elapsed since the last YOLOX run.
    * Predicted bbox is ``None`` (track lost).
    * Predicted bbox area < 25% of expected (drastic size change, probable
      track switch or person left frame).
    """
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
