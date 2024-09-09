from copy import deepcopy
from typing import Dict
import numpy as np

from skellytracker.trackers.base_tracker.base_recorder import BaseRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.yolo_tracker.yolo_model_info import YOLOModelInfo


class YOLORecorder(BaseRecorder):
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        self.recorded_objects.append(deepcopy(tracked_objects["tracked_person"]))

    def process_tracked_objects(self, **kwargs) -> np.ndarray:
        self.recorded_objects_array = np.zeros(
            (len(self.recorded_objects), YOLOModelInfo.num_tracked_points, 3)
        )

        for i, recorded_object in enumerate(self.recorded_objects):
            for j in range(YOLOModelInfo.num_tracked_points):
                self.recorded_objects_array[i, j, 0] = recorded_object.extra[
                    "landmarks"
                ][0, j, 0]
                self.recorded_objects_array[i, j, 1] = recorded_object.extra[
                    "landmarks"
                ][0, j, 1]
                self.recorded_objects_array[i, j, 2] = np.nan

        return self.recorded_objects_array
