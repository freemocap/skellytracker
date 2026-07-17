from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.data_primitives.keypoints import Keypoints


@dataclass
class BBoxSmoothingState:
    """EMA state for smoothing bounding box position and size across frames."""

    smooth_bbox: BoundingBox | None = None
    last_detection_frame: int | None = None


@dataclass
class KeypointSmoothingState:
    """Filter state for smoothing keypoint coordinates across frames.

    Used by both the one-euro filter and the Kalman filter.  Fields that are
    not relevant to the active filter are left as None.

    One-euro fields:  x_prev, y_prev, dx_prev, dy_prev
    Kalman fields:    x_prev, y_prev (position estimates), vx_est, vy_est,
                      Px (N,2,2), Py (N,2,2)
    Shared:           frames_since_detection
    """

    # --- position estimates (both filters) ---
    x_prev: NDArray[np.float64] | None = None   # (N,) last filtered x values
    y_prev: NDArray[np.float64] | None = None   # (N,) last filtered y values

    # --- one-euro derivative state ---
    dx_prev: NDArray[np.float64] | None = None  # (N,) last x derivative estimates
    dy_prev: NDArray[np.float64] | None = None  # (N,) last y derivative estimates

    # --- Kalman velocity and covariance state ---
    vx_est: NDArray[np.float64] | None = None   # (N,) x velocity estimates
    vy_est: NDArray[np.float64] | None = None   # (N,) y velocity estimates
    Px: NDArray[np.float64] | None = None       # (N, 2, 2) x-axis covariance matrices
    Py: NDArray[np.float64] | None = None       # (N, 2, 2) y-axis covariance matrices

    # --- shared gap-fill counter ---
    frames_since_detection: NDArray[np.int32] | None = None  # (N,) frames since last trusted detection


@dataclass
class StageState:
    """Temporal state for a single DetectionStage."""

    bbox_state: BBoxSmoothingState = field(default_factory=BBoxSmoothingState)
    keypoint_states: list[KeypointSmoothingState] = field(default_factory=list)
    child_states: dict[str, StageState] = field(default_factory=dict)
    last_keypoints: Keypoints | None = None
    # Consecutive zero-valid-keypoint frames per keypoint_detectors index, used by
    # KeypointResetPolicy to detect and recover from stuck tracker state.
    consecutive_misses: list[int] = field(default_factory=list)


@dataclass
class TrackerState:
    """All temporal state for a Tracker, passed in and returned per frame.

    The Tracker itself is stateless between calls; all smoothing history
    lives here. Serializable for resumption after a pause.
    """

    stage_states: dict[str, StageState] = field(default_factory=dict)

    @staticmethod
    def empty() -> TrackerState:
        return TrackerState()
