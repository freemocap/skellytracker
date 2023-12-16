from typing import Dict
from copy import deepcopy
import numpy as np

from skellytracker.trackers.base_tracker.base_recorder import BaseRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject


class BrightestPointRecorder(BaseRecorder):
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        self.recorded_objects.append(deepcopy(tracked_objects["brightest_point"]))

    def process_tracked_objects(self, **kwargs) -> np.ndarray:
        self.recorded_objects_array = np.zeros((len(self.recorded_objects), 1, 3))
        for i, recorded_object in enumerate(self.recorded_objects):
            self.recorded_objects_array[i, 0, 0] = recorded_object.pixel_x
            self.recorded_objects_array[i, 0, 1] = recorded_object.pixel_y
            self.recorded_objects_array[i, 0, 2] = recorded_object.depth_z

        return self.recorded_objects_array
