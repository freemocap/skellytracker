"""Array mapping across frames; missing dependencies never become observations."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MappedLandmarkBatch:
    landmark_names: tuple[str, ...]
    positions: np.ndarray
    present: np.ndarray


def map_tracker_batch(
    mapping, *, tracker_names: tuple[str, ...], positions: np.ndarray
) -> MappedLandmarkBatch:
    """One-shot mapping; use PreparedTrackerMapping to reuse a configured plan."""
    from skellytracker.core.io.prepared_tracker_mapping import PreparedTrackerMapping

    return PreparedTrackerMapping.prepare(mapping, tracker_names=tracker_names).apply(
        tracker_names=tracker_names, positions=positions
    )
