from __future__ import annotations

from dataclasses import dataclass, field

from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.tracker.tracker_state import StageState


@dataclass
class PersonTrackState:
    """Temporal state for one tracked person, persisted across frames.

    Wraps the same per-stage smoothing/reset-policy state a single-person
    Tracker uses (StageState), keyed by stage name — one PersonTrackState is
    functionally a single-subject TrackerState plus multi-person bookkeeping
    (identity, match history, staleness).
    """

    track_id: int
    stage_states: dict[str, StageState] = field(default_factory=dict)
    last_bbox: BoundingBox | None = None
    last_keypoints: Keypoints | None = None
    hits: int = 0
    time_since_update: int = 0
    age: int = 0
    center_velocity: tuple[float, float] = (0.0, 0.0)

    def confirmed(self, min_hits: int) -> bool:
        return self.hits >= min_hits

    def predicted_bbox(self) -> BoundingBox | None:
        """Constant-velocity prediction used before Hungarian association."""
        if self.last_bbox is None:
            return None
        dx, dy = self.center_velocity
        return BoundingBox(
            x1=self.last_bbox.x1 + dx,
            y1=self.last_bbox.y1 + dy,
            x2=self.last_bbox.x2 + dx,
            y2=self.last_bbox.y2 + dy,
            confidence=self.last_bbox.confidence,
        )

    def update_motion(self, bbox: BoundingBox) -> None:
        if self.last_bbox is not None:
            old_x, old_y = self.last_bbox.center
            new_x, new_y = bbox.center
            measured = (new_x - old_x, new_y - old_y)
            self.center_velocity = (
                0.65 * self.center_velocity[0] + 0.35 * measured[0],
                0.65 * self.center_velocity[1] + 0.35 * measured[1],
            )
