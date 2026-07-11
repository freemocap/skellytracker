from __future__ import annotations

from dataclasses import replace

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
