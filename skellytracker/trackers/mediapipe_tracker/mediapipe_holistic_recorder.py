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
                for tracked_object_name in MediapipeModelInfo.mediapipe_tracked_object_names
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
                MediapipeModelInfo.num_tracked_points_total,
                3,
            )
        )

        for i, recorded_object_list in enumerate(self.recorded_objects):
            frame_data = {
                name: np.full((self.num_tracked_points_by_name(name), 3), np.nan)
                for name in MediapipeModelInfo.mediapipe_tracked_object_names
            }
            for recorded_object in recorded_object_list:
                if recorded_object.extra["landmarks"] is not None:
                    for j, landmark_data in enumerate(
                        recorded_object.extra["landmarks"].landmark
                    ):
                        frame_data[recorded_object.object_id][j, 0] = (
                            landmark_data.x * image_size[0]
                        )
                        frame_data[recorded_object.object_id][j, 1] = (
                            landmark_data.y * image_size[1]
                        )
                        frame_data[recorded_object.object_id][j, 2] = (
                            landmark_data.z * image_size[0]
                        )  # multiply depth by image width, per MediaPipe documentation

            for name in MediapipeModelInfo.mediapipe_tracked_object_names:
                if name not in frame_data:
                    frame_data[name] = np.full(
                        self.num_tracked_points_by_name(name), np.nan
                    )

            self.recorded_objects_array[i] = np.concatenate(
                # this order matters, do not change
                (
                    frame_data["pose_landmarks"],
                    frame_data["right_hand_landmarks"],
                    frame_data["left_hand_landmarks"],
                    frame_data["face_landmarks"],
                ),
                axis=0,
            )

        return self.recorded_objects_array

    def num_tracked_points_by_name(self, name: str) -> int:
        if name == "pose_landmarks":
            num_tracked_points = MediapipeModelInfo.num_tracked_points_body
        elif name == "face_landmarks":
            num_tracked_points = MediapipeModelInfo.num_tracked_points_face
        elif name == "left_hand_landmarks":
            num_tracked_points = MediapipeModelInfo.num_tracked_points_left_hand
        elif name == "right_hand_landmarks":
            num_tracked_points = MediapipeModelInfo.num_tracked_points_right_hand
        else:
            raise ValueError(
                f"Invalid tracked object ID for mediapipe holistic tracker: {name}"
            )

        return num_tracked_points
