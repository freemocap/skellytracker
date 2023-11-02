from typing import Dict

import numpy as np

from skelly_tracker.trackers.base_tracker.base_recorder import BaseRecorder
from skelly_tracker.trackers.base_tracker.tracked_object import TrackedObject
from skelly_tracker.trackers.mediapipe_tracker.mediapipe_model_info import (
    MediapipeModelInfo,
)


class MediapipeHolisticRecorder(BaseRecorder):
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        self.recorded_objects.append(
            [
                tracked_objects[tracked_object_name]
                for tracked_object_name in MediapipeModelInfo.mediapipe_tracked_object_names
            ]
        )

    def process_tracked_objects(self) -> np.ndarray:
        self.recorded_objects_array = np.zeros(
            (len(self.recorded_objects), MediapipeModelInfo.num_tracked_points_total, 3)
        )

        for i, recorded_object_list in enumerate(self.recorded_objects):
            landmark_number = 0
            for recorded_object in recorded_object_list:
                for landmark_data in recorded_object.extra["landmarks"].landmark:
                    self.recorded_objects_array[
                        i, landmark_number, 0
                    ] = landmark_data.x  # this needs to be * image width
                    self.recorded_objects_array[
                        i, landmark_number, 1
                    ] = landmark_data.y  # this needs to be * image height
                    self.recorded_objects_array[
                        i, landmark_number, 2
                    ] = (
                        landmark_data.z
                    )  # this needs to be * image width # * image width per mediapipe docs
                    landmark_number += 1

        return self.recorded_objects_array
