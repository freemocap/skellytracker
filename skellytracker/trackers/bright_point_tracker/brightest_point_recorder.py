from typing import Dict
import numpy as np

from skellytracker.trackers.base_tracker.base_recorder import BaseRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject


class BrightestPointRecorder(BaseRecorder):
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        self.recorded_objects.append(
            [
                (tracked_object.pixel_x, tracked_object.pixel_y)
                for tracked_object in tracked_objects.values()
                if "brightest_point" in tracked_object.object_id
            ]
        )

    def process_tracked_objects(self, **kwargs) -> np.ndarray:
        num_frames = len(self.recorded_objects)
        num_points = len(self.recorded_objects[0]) if num_frames > 0 else 0

        self.recorded_objects_array = np.zeros((num_frames, num_points, 2))
        for i, recorded_object in enumerate(self.recorded_objects):
            for j, (pixel_x, pixel_y) in enumerate(recorded_object):
                self.recorded_objects_array[i, j, 0] = pixel_x
                self.recorded_objects_array[i, j, 1] = pixel_y

        return self.recorded_objects_array
