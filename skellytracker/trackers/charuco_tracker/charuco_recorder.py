from copy import deepcopy
from typing import Dict
import numpy as np

from skellytracker.trackers.base_tracker.base_recorder import BaseRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject


class CharucoRecorder(BaseRecorder):
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        self.recorded_objects.append(
            [deepcopy(tracked_object) for tracked_object in tracked_objects.values()]
        )

    def process_tracked_objects(self, **kwargs) -> np.ndarray:
        self.recorded_objects_array = np.array(
            [
                [[tracked_object.pixel_x, tracked_object.pixel_y] for tracked_object in tracked_object_list]
                for tracked_object_list in self.recorded_objects
            ]
        )

        return self.recorded_objects_array
