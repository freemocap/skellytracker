from copy import deepcopy
from typing import Dict
import numpy as np

from skellytracker.trackers.base_tracker.base_recorder import BaseRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject


class YOLOObjectRecorder(BaseRecorder):
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        self.recorded_objects.append(deepcopy(tracked_objects["object"]))

    def process_tracked_objects(self, **kwargs) -> np.ndarray:
        self.recorded_objects_array = np.zeros((len(self.recorded_objects), 4))
        for i, recorded_object in enumerate(self.recorded_objects):
            self.recorded_objects_array[i, :] = recorded_object.extra["boxes_xyxy"]

        return self.recorded_objects_array
