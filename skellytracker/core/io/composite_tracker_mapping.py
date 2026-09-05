"""Compose disjoint landmark mappings without changing their evaluation rules."""

from dataclasses import dataclass

import numpy as np

from skellytracker.core.io.tracker_mapping import TrackerMapping, TrackerMappingSnapshot


@dataclass(frozen=True, slots=True)
class CompositeTrackerMapping:
    mappings: tuple[TrackerMapping, ...]

    def __post_init__(self) -> None:
        if not self.mappings:
            raise ValueError("Composite mapping requires at least one mapping")
        names = [name for mapping in self.mappings for name in mapping.landmark_names]
        if len(set(names)) != len(names):
            raise ValueError("Composite mappings must produce disjoint landmark names")

    def apply(self, tracker_positions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            name: value
            for mapping in self.mappings
            for name, value in mapping.apply(tracker_positions).items()
        }

    @property
    def directly_measured_landmark_names(self) -> frozenset[str]:
        return frozenset(
            name
            for mapping in self.mappings
            for name in mapping.directly_measured_landmark_names
        )

    def mapping_snapshots(self) -> tuple[TrackerMappingSnapshot, ...]:
        return tuple(
            snapshot
            for mapping in self.mappings
            for snapshot in mapping.mapping_snapshots()
        )
