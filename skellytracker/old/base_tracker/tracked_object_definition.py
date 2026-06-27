"""
TrackedObjectDefinition — the schema of named points + connections that a
tracker produces, loaded from YAML.

This is the single source of truth for tracker metadata. Detectors construct
PointClouds whose names come from a definition. Annotators draw connections
by resolving the definition's name-pairs to indices into that PointCloud.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from skellytracker.old.base_tracker.point_cloud import PointCloud


class TrackedObjectDefinition(BaseModel):
    """A named set of tracked points plus the connections between them."""

    model_config = ConfigDict(frozen=True)

    name: str
    tracker_type: str
    landmark_schema: str = "unknown"
    tracked_points: tuple[str, ...]
    connections: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="after")
    def _validate_connections_reference_known_points(self) -> "TrackedObjectDefinition":
        known = set(self.tracked_points)
        for a, b in self.connections:
            if a not in known or b not in known:
                missing = [n for n in (a, b) if n not in known]
                raise ValueError(
                    f"TrackedObjectDefinition {self.name!r}: connection {(a, b)} "
                    f"references unknown point(s) {missing}"
                )
        if len(self.tracked_points) != len(set(self.tracked_points)):
            raise ValueError(
                f"TrackedObjectDefinition {self.name!r}: tracked_points contains duplicates"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "TrackedObjectDefinition":
        """Load a single definition YAML.

        If the YAML has a `composed_of` key, delegate to composition loading.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "composed_of" in data:
            return cls._from_composition_data(data=data, base_dir=path.parent)

        return cls(
            name=data["name"],
            tracker_type=data["tracker_type"],
            landmark_schema=data.get("landmark_schema", "unknown"),
            tracked_points=tuple(data.get("tracked_points", ())),
            connections=tuple(tuple(pair) for pair in data.get("connections", ())),
        )

    @classmethod
    def _from_composition_data(
        cls, data: dict[str, Any], base_dir: Path
    ) -> "TrackedObjectDefinition":
        """Flatten a composition YAML into a single definition.

        Each entry in `composed_of` is `{path: <relative>, prefix: <str>}`. Child
        definitions are loaded recursively; their point names are prefixed and
        their connections are remapped to the prefixed names.
        """
        all_points: list[str] = []
        all_connections: list[tuple[str, str]] = []

        for entry in data["composed_of"]:
            child_path = (base_dir / entry["path"]).resolve()
            prefix: str = entry.get("prefix", "")
            child = cls.from_yaml(child_path)

            prefixed = {p: f"{prefix}{p}" for p in child.tracked_points}
            all_points.extend(prefixed.values())
            all_connections.extend((prefixed[a], prefixed[b]) for a, b in child.connections)

        return cls(
            name=data["name"],
            tracker_type=data["tracker_type"],
            landmark_schema=data.get("landmark_schema", "composite"),
            tracked_points=tuple(all_points),
            connections=tuple(all_connections),
        )

    def with_prefix(self, prefix: str) -> "TrackedObjectDefinition":
        """Return a copy with `prefix` prepended to every tracked-point name."""
        if not prefix:
            return self
        renamed = {p: f"{prefix}{p}" for p in self.tracked_points}
        return self.model_copy(update={
            "tracked_points": tuple(renamed.values()),
            "connections": tuple((renamed[a], renamed[b]) for a, b in self.connections),
        })

    def concatenate(self, other: "TrackedObjectDefinition", *, name: str | None = None, tracker_type: str | None = None) -> "TrackedObjectDefinition":
        """Concatenate two definitions in order. Names must be disjoint."""
        overlap = set(self.tracked_points) & set(other.tracked_points)
        if overlap:
            raise ValueError(f"Cannot concatenate: overlapping point names {sorted(overlap)}")
        return TrackedObjectDefinition(
            name=name or f"{self.name}+{other.name}",
            tracker_type=tracker_type or self.tracker_type,
            tracked_points=self.tracked_points + other.tracked_points,
            connections=self.connections + other.connections,
        )

    def connection_indices(self) -> tuple[tuple[int, int], ...]:
        """Resolve name-pairs to (start_idx, end_idx) pairs.

        Use for cv2.line drawing against an (N, ...) coordinate array whose
        rows are in tracked_points order.
        """
        idx_of = {name: i for i, name in enumerate(self.tracked_points)}
        return tuple((idx_of[a], idx_of[b]) for a, b in self.connections)

    def index_of(self, name: str) -> int:
        """Row index for a tracked point name. Raises KeyError if missing."""
        return self.tracked_points.index(name)

    def empty_point_cloud(self) -> PointCloud:
        """Factory for an empty PointCloud sized/named per this definition."""
        return PointCloud.empty(self.tracked_points)

    @property
    def num_tracked_points(self) -> int:
        return len(self.tracked_points)
