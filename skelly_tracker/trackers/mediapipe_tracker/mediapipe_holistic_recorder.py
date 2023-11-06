from copy import deepcopy
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
                deepcopy(tracked_objects[tracked_object_name])
                for tracked_object_name in MediapipeModelInfo.mediapipe_tracked_object_names.value
            ]
        )

    def process_tracked_objects(self, **kwargs) -> np.ndarray:
        image_size = kwargs.get("image_size")
        if image_size is None:
            raise ValueError("image_size must be provided to process tracked objects")
        self.recorded_objects_array = np.zeros(
            (
                len(self.recorded_objects),
                MediapipeModelInfo.num_tracked_points_total.value,
                3,
            )
        )
        print(len(self.recorded_objects))
        for i, recorded_object_list in enumerate(self.recorded_objects):
            landmark_number = 0
            for recorded_object in recorded_object_list:
                if recorded_object.extra["landmarks"] is not None:
                    for landmark_data in recorded_object.extra["landmarks"].landmark:
                        self.recorded_objects_array[i, landmark_number, 0] = (
                            landmark_data.x * image_size[0]
                        )
                        self.recorded_objects_array[i, landmark_number, 1] = (
                            landmark_data.y * image_size[1]
                        )
                        self.recorded_objects_array[i, landmark_number, 2] = (
                            landmark_data.z * image_size[0]
                        )  # * image width per mediapipe docs
                        landmark_number += 1
                else:
                    if recorded_object.object_id == "pose_landmarks":
                        number = MediapipeModelInfo.num_tracked_points_body.value
                    elif recorded_object.object_id == "face_landmarks":
                        number = MediapipeModelInfo.num_tracked_points_face.value
                    elif recorded_object.object_id == "left_hand_landmarks":
                        number = MediapipeModelInfo.num_tracked_points_left_hand.value
                    else:
                        number = MediapipeModelInfo.num_tracked_points_right_hand.value
                    for _ in range(number):
                        self.recorded_objects_array[i, landmark_number, :] = np.NaN
                        landmark_number += 1

        return self.recorded_objects_array
