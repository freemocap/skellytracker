from copy import deepcopy
from typing import Dict
import numpy as np

from skellytracker.trackers.base_tracker.base_recorder import BaseRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.mediapipe_tracker.mediapipe_model_info import (
    MediapipeModelInfo,
)


class MediapipeHolisticRecorder(BaseRecorder):
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        self.recorded_objects.append(
            [
                deepcopy(tracked_objects[tracked_object_name])
                for tracked_object_name in MediapipeModelInfo.tracked_object_names
            ]
        )

    def process_tracked_objects(self, **kwargs) -> np.ndarray:
        image_size = kwargs.get("image_size")
        if image_size is None:
            raise ValueError(
                f"image_size must be provided to process tracked objects from {__class__.__name__}"
            )
        self.recorded_objects_array = np.zeros(
            (
                len(self.recorded_objects),
                MediapipeModelInfo.num_tracked_points,
                3,
            )
        )

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
                        number = MediapipeModelInfo.num_tracked_points_body
                    elif recorded_object.object_id == "face_landmarks":
                        number = MediapipeModelInfo.num_tracked_points_face
                    elif recorded_object.object_id == "left_hand_landmarks":
                        number = MediapipeModelInfo.num_tracked_points_left_hand
                    else:
                        number = MediapipeModelInfo.num_tracked_points_right_hand
                    for _ in range(number):
                        self.recorded_objects_array[i, landmark_number, :] = np.nan
                        landmark_number += 1

        return self.recorded_objects_array
