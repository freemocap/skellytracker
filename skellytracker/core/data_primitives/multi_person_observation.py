from __future__ import annotations

from dataclasses import dataclass, field

from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.data_primitives.observation import Observation


@dataclass
class MultiPersonObservation:
    """Per-frame output of a MultiPersonTracker.

    people is keyed by track_id, stable across frames for as long as the
    underlying person is tracked continuously (see PersonTrackState / the
    track_association module for the matching that assigns these IDs).
    """

    frame_number: int
    image_size: tuple[int, int]  # (height, width) in pixels
    people: dict[int, Observation] = field(default_factory=dict)

    def to_keypoints_by_track(self) -> dict[int, Keypoints]:
        return {track_id: obs.to_keypoints() for track_id, obs in self.people.items()}
