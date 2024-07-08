from copy import deepcopy
from typing import Dict
import numpy as np

from skellytracker.trackers.base_tracker.base_recorder import BaseRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.mediapipe_blendshape_tracker.mediapipe_blendshape_model_info import (
    MediapipeBlendshapeModelInfo,
)


class MediapipeBlendshapeRecorder(BaseRecorder):
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        self.recorded_objects.append([deepcopy(tracked_objects["face"])])

    def process_tracked_objects(self, **kwargs) -> np.ndarray:
        image_size = kwargs.get("image_size")
        if image_size is None:
            raise ValueError(
                f"image_size must be provided to process tracked objects from {__class__.__name__}"
            )
        self.recorded_objects_array = np.full(
            (
                len(self.recorded_objects),
                MediapipeBlendshapeModelInfo.num_tracked_points,
                1,
            ),
            np.nan,
        )

        for i, tracked_objects in enumerate(self.recorded_objects):
            if blendshapes := tracked_objects[0].extra.get("blendshapes", None):
                self.recorded_objects_array[i, :, 0] = blendshapes

        return self.recorded_objects_array
