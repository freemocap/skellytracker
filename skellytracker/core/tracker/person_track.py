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

    def confirmed(self, min_hits: int) -> bool:
        return self.hits >= min_hits
