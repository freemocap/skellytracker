from typing import Dict

import numpy as np

from skelly_tracker.trackers.base_tracker.base_recorder import BaseRecorder
from skelly_tracker.trackers.base_tracker.tracked_object import TrackedObject
from skelly_tracker.trackers.yolo_tracker.yolo_model_info import NUM_TRACKED_POINTS_YOLO


class YOLORecorder(BaseRecorder):
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        self.recorded_objects.append(tracked_objects["tracked_person"])

    def process_tracked_objects(self) -> np.ndarray:
        self.recorded_objects_array = np.zeros(
            (len(self.recorded_objects), NUM_TRACKED_POINTS_YOLO, 3)
        )
        print(self.recorded_objects[-1].extra["landmarks"])
        for i, recorded_object in enumerate(self.recorded_objects):
            for j in range(NUM_TRACKED_POINTS_YOLO):
                self.recorded_objects_array[i, j, 0] = recorded_object.extra[
                    "landmarks"
                ][0, j, 0]
                self.recorded_objects_array[i, j, 1] = recorded_object.extra[
                    "landmarks"
                ][0, j, 1]
                self.recorded_objects_array[i, j, 2] = np.NaN

        return self.recorded_objects_array
