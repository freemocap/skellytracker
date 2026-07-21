from dataclasses import dataclass, field

import numpy as np

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation
from skellytracker.trackers.base_tracker.point_cloud import PointCloud


@dataclass(slots=True)
class BrightPatch:
    area: float
    centroid_x: int
    centroid_y: int


@dataclass(slots=True)
class BrightestPointObservation(BaseObservation):
    tracker_type: str = field(default="bright_point_tracker", init=False)
    frame_number: int = 0
    points: PointCloud = field(default_factory=lambda: PointCloud.empty(()))
    bright_patches: list[BrightPatch | None] = field(default_factory=list)

    @classmethod
    def from_detection_results(cls, frame_number: int, bright_patches: list[BrightPatch | None]) -> "BrightestPointObservation":
        n = len(bright_patches)
        names = tuple(f"bright_patch_{i + 1}" for i in range(n))
        xyz = np.full((n, 3), np.nan)
        visibility = np.zeros(n)

        for i, patch in enumerate(bright_patches):
            if patch is not None:
                xyz[i] = (patch.centroid_x, patch.centroid_y, 0.0)
                visibility[i] = 1.0

        cloud = PointCloud(names=names, xyz=xyz, visibility=visibility)

        return cls(
            frame_number=frame_number,
            points=cloud,
            bright_patches=bright_patches,
        )
