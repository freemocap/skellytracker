"""Compose disjoint landmark mappings without changing their evaluation rules."""

from dataclasses import dataclass

import numpy as np

from skellytracker.core.io.tracker_mapping import MappedLandmarkEvidence, TrackerMapping, TrackerMappingSnapshot


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

    def apply_with_quality(
        self, *, tracker_positions: dict[str, np.ndarray], tracker_quality: dict[str, float],
    ) -> MappedLandmarkEvidence:
        positions: dict[str, np.ndarray] = {}
        quality: dict[str, float] = {}
        for mapping in self.mappings:
            evidence = mapping.apply_with_quality(
                tracker_positions=tracker_positions, tracker_quality=tracker_quality,
            )
            duplicates = positions.keys() & evidence.positions.keys()
            if duplicates:
                raise ValueError(f"Composite mappings produced duplicate landmarks: {sorted(duplicates)}")
            positions.update(evidence.positions)
            quality.update(evidence.quality)
        return MappedLandmarkEvidence(positions=positions, quality=quality)

    def mapping_snapshots(self) -> tuple[TrackerMappingSnapshot, ...]:
        return tuple(
            snapshot
            for mapping in self.mappings
            for snapshot in mapping.mapping_snapshots()
        )
