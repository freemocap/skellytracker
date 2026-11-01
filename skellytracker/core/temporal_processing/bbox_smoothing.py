from __future__ import annotations

from dataclasses import replace
import numpy as np
from skellytracker.core.temporal_processing.one_euro_array import OneEuroWorkspace, one_euro_step
from skellytracker.core.temporal_processing.temporal_processing_config import BBoxSmoothingConfig

from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.tracker.tracker_state import BBoxSmoothingState


def apply_bbox_ema(
    raw_bbox: BoundingBox,
    state: BBoxSmoothingState,
    alpha: float,
) -> tuple[BoundingBox, BBoxSmoothingState]:
    """Apply EMA smoothing to a bounding box.

    alpha=0 means no history (raw values pass through).
    alpha=1 means full history (bbox never updates). Typical range: 0.3–0.7.
    """
    if state.smooth_bbox is None:
        smooth = raw_bbox
    else:
        prev = state.smooth_bbox
        prev_cx, prev_cy = prev.center
        raw_cx, raw_cy = raw_bbox.center

        cx = alpha * prev_cx + (1.0 - alpha) * raw_cx
        cy = alpha * prev_cy + (1.0 - alpha) * raw_cy
        w = alpha * prev.width + (1.0 - alpha) * raw_bbox.width
        h = alpha * prev.height + (1.0 - alpha) * raw_bbox.height
        smooth = BoundingBox.from_center_size(cx, cy, w, h, confidence=raw_bbox.confidence)

    return smooth, replace(state, smooth_bbox=smooth)


def apply_bbox_one_euro(raw_bbox: BoundingBox | None, state: BBoxSmoothingState,
                        config: BBoxSmoothingConfig, dt: float | None):
    """Filter corner coordinates using reusable tensors; never invent a missing crop."""
    if raw_bbox is None:
        return None, replace(state, smooth_bbox=None, corner_workspace=None, raw_corners=None)
    work = state.corner_workspace
    raw = state.raw_corners
    reset = work is None or dt is None or dt > config.reset_after_seconds
    if work is None:
        work = OneEuroWorkspace.create((2, 2))
        raw = np.empty((2, 2), dtype=np.float64)
    raw[0] = (raw_bbox.x1, raw_bbox.y1)
    raw[1] = (raw_bbox.x2, raw_bbox.y2)
    if not np.isfinite(raw).all():
        return None, replace(state, smooth_bbox=None, corner_workspace=None, raw_corners=None)
    if reset:
        np.copyto(work.filtered_values, raw)
        work.derivative_values.fill(0.0)
    else:
        one_euro_step(raw, work.filtered_values, work.derivative_values, dt=dt,
            min_cutoff=config.min_cutoff, beta=config.beta, d_cutoff=config.d_cutoff, workspace=work)
    # Different adaptive cutoffs can make corners cross during abrupt resizing.
    work.filtered_values.sort(axis=0)
    smooth = BoundingBox(x1=float(work.filtered_values[0, 0]), y1=float(work.filtered_values[0, 1]),
                         x2=float(work.filtered_values[1, 0]), y2=float(work.filtered_values[1, 1]),
                         confidence=raw_bbox.confidence)
    return smooth, replace(state, smooth_bbox=smooth, corner_workspace=work, raw_corners=raw)
