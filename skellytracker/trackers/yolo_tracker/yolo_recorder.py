from copy import deepcopy
from typing import Dict
import numpy as np

from skellytracker.trackers.base_tracker.base_recorder import BaseRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.yolo_tracker.yolo_model_info import YOLOModelInfo


class YOLORecorder(BaseRecorder):
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        self.recorded_objects.append(
            [
                deepcopy(tracked_object)
                for tracked_object in tracked_objects.values()
                if "tracked_person" in tracked_object.object_id
            ]
        )

    def process_tracked_objects(self, **kwargs) -> np.ndarray:
        num_tracked_points = YOLOModelInfo.num_tracked_points * len(self.recorded_objects[-1])
        recorded_object_arrays = []
        for i, recorded_object in enumerate(self.recorded_objects):
            recorded_objects_arrays = []
            for tracked_object in recorded_object:
                tracked_object_data_array = np.zeros((len(self.recorded_objects), num_tracked_points, 2))
                for j in range(YOLOModelInfo.num_tracked_points):
                    tracked_object_data_array[i, j, 0] = tracked_object.extra[
                        "landmarks"
                    ][0, j, 0]
                    tracked_object_data_array[i, j, 1] = tracked_object.extra[
                        "landmarks"
                    ][0, j, 1]

                recorded_objects_arrays.append(tracked_object_data_array)

            recorded_object_data_array = np.concatenate(recorded_objects_arrays, axis=1)

            recorded_object_arrays.append(recorded_object_data_array)

        self.recorded_objects_array = np.concatenate(recorded_object_arrays, axis=0)
        return self.recorded_objects_array
