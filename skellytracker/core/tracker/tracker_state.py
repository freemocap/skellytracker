from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class BBoxSmoothingState:
    """EMA state for smoothing bounding box position and size across frames."""

    smooth_center: tuple[float, float] | None = None
    smooth_diagonal: float | None = None
    alpha: float = 0.5  # EMA decay: 0 = no smoothing, 1 = no history


@dataclass
class KeypointSmoothingState:
    """One euro filter state for smoothing keypoint coordinates across frames.

    One entry per point in the associated KeypointDetector's schema.
    """

    x_prev: NDArray[np.float64] | None = None   # (N,) last filtered x values
    y_prev: NDArray[np.float64] | None = None   # (N,) last filtered y values
    dx_prev: NDArray[np.float64] | None = None  # (N,) last derivative estimates
    dy_prev: NDArray[np.float64] | None = None


@dataclass
class StageState:
    """Temporal state for a single DetectionStage."""

    bbox_state: BBoxSmoothingState = field(default_factory=BBoxSmoothingState)
    keypoint_states: list[KeypointSmoothingState] = field(default_factory=list)
    child_states: dict[str, StageState] = field(default_factory=dict)


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
