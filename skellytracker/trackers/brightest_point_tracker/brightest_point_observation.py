import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation, TrackerTypeString, \
    TrackedPointIdString, TrackedPoint2dArray


class BrightPatch(BaseModel):
    area: float
    centroid_x: int
    centroid_y: int


class BrightestPointObservation(BaseObservation):
    bright_patches: list[BrightPatch | None]
    tracker_type:TrackerTypeString = 'bright_point_tracker'

    @classmethod
    def from_detection_results(cls, frame_number: int, bright_patches: list[BrightPatch | None]):
        return cls(frame_number=frame_number, bright_patches=bright_patches)

    def to_2d_array(self) -> NDArray[Shape["* bright_patches, 2 pxpy"], float]:
        array = np.full((len(self.bright_patches), 2), np.nan)
        for patch_index, patch in enumerate(self.bright_patches):
            if patch is None:
                continue
            array[patch_index, 0] = patch.centroid_x
            array[patch_index, 1] = patch.centroid_y
        return array

    def to_tracked_points(cls) -> dict[TrackedPointIdString, TrackedPoint2dArray]:
        points = {}
        for i, patch in enumerate(cls.bright_patches):
            if patch is None:
                continue
            point_id = f"bright_patch_{i+1}"
            points[point_id] = np.array([patch.centroid_x, patch.centroid_y])
        return points
