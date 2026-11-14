"""One camera-local hand track. Handedness describes the hand, not its identity."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from skellytracker.core.data_primitives.keypoints import Keypoints

HandCount = Literal["one", "two", "unknown"]
HandSide = Literal["left", "right"]


@dataclass(frozen=True)
class TrackedHand:
    track_id: int
    handedness: HandSide
    keypoints: Keypoints

    def sided_keypoints(self) -> Keypoints:
        """Expose this instance in the anatomical mapping's sided input schema."""
        names = _sided_names(self.handedness, self.keypoints.names)
        return Keypoints(names, self.keypoints.xyz, self.keypoints.visibility)


@lru_cache(maxsize=2)
def _sided_names(side: HandSide, names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{side}_hand_{name}" for name in names)
