import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel
from skellytracker.trackers.base_tracker.base_tracker import BaseObservation


class BrightPatch(BaseModel):
    area: float
    centroid_x: int
    centroid_y: int


class BrightestPointObservation(BaseObservation):
    bright_patches: list[BrightPatch]

    @classmethod
    def from_detection_results(cls, frame_number: int, bright_patches: list[BrightPatch]):
        return cls(frame_number=frame_number, bright_patches=bright_patches)

    def to_array(self) -> NDArray[Shape["* bright_patches, 2 pxpy"], float]:
        array = np.full((len(self.bright_patches), 2), np.nan)
        for patch_index, patch in enumerate(self.bright_patches):
            array[patch_index, 0] = patch.centroid_x
            array[patch_index, 1] = patch.centroid_y
        return array
