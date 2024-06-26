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
        num_frames = len(self.recorded_objects)
        num_tracked_objects = len(self.recorded_objects[0])
        num_points = YOLOModelInfo.num_tracked_points

        self.recorded_objects_array = np.zeros(
            (num_frames, num_tracked_objects, num_points, 2)
        )

        for i, recorded_object in enumerate(self.recorded_objects):
            for j, tracked_object in enumerate(recorded_object):
                self.recorded_objects_array[i, j, :, :] = tracked_object.extra["landmarks"]

        self.recorded_objects_array = self.recorded_objects_array.reshape(
            num_frames, num_tracked_objects * num_points, 2
        )
        return self.recorded_objects_array
