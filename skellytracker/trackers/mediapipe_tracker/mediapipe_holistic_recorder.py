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
            frame_data = {}
            for recorded_object in recorded_object_list:
                if recorded_object.object_id == "pose_landmarks":
                    num_tracked_points = MediapipeModelInfo.num_tracked_points_body
                elif recorded_object.object_id == "face_landmarks":
                    num_tracked_points = MediapipeModelInfo.num_tracked_points_face
                elif recorded_object.object_id == "left_hand_landmarks":
                    num_tracked_points = MediapipeModelInfo.num_tracked_points_left_hand
                elif recorded_object.object_id == "right_hand_landmarks":
                    num_tracked_points = (
                        MediapipeModelInfo.num_tracked_points_right_hand
                    )
                else:
                    raise ValueError(
                        f"Invalid tracked object ID for mediapipe holistic tracker: {recorded_object.object_id}"
                    )
                tracked_object_array = np.zeros((num_tracked_points, 3))
                if recorded_object.extra["landmarks"] is not None:
                    for j, landmark_data in enumerate(
                        recorded_object.extra["landmarks"].landmark
                    ):
                        tracked_object_array[j, 0] = landmark_data.x * image_size[0]
                        tracked_object_array[j, 1] = landmark_data.y * image_size[1]
                        tracked_object_array[j, 2] = (
                            landmark_data.z * image_size[0]
                        )  # multiply depth by image width, per MediaPipe documentation
                else:
                    tracked_object_array[:] = np.nan

                frame_data[recorded_object.object_id] = tracked_object_array
                print(tracked_object_array.shape)

            self.recorded_objects_array[i] = np.concatenate(
                # this order matters, do not change
                (
                    frame_data["pose_landmarks"],
                    frame_data["face_landmarks"],
                    frame_data["left_hand_landmarks"],
                    frame_data["right_hand_landmarks"],
                ),
                axis=0,
            )

        return self.recorded_objects_array
        
    # def process_tracked_objects(self, **kwargs) -> np.ndarray:
    #     image_size = kwargs.get("image_size")
    #     if image_size is None:
    #         raise ValueError(f"image_size must be provided to process tracked objects from {__class__.__name__}")
    #     self.recorded_objects_array = np.zeros(
    #         (
    #             len(self.recorded_objects),
    #             MediapipeModelInfo.num_tracked_points_total,
    #             3,
    #         )
    #     )

    #     for i, recorded_object_list in enumerate(self.recorded_objects):
    #         landmark_number = 0
    #         for recorded_object in recorded_object_list:
    #             if recorded_object.extra["landmarks"] is not None:
    #                 for landmark_data in recorded_object.extra["landmarks"].landmark:
    #                     self.recorded_objects_array[i, landmark_number, 0] = (
    #                         landmark_data.x * image_size[0]
    #                     )
    #                     self.recorded_objects_array[i, landmark_number, 1] = (
    #                         landmark_data.y * image_size[1]
    #                     )
    #                     self.recorded_objects_array[i, landmark_number, 2] = (
    #                         landmark_data.z * image_size[0]
    #                     )  # * image width per mediapipe docs
    #                     landmark_number += 1
    #             else:
    #                 if recorded_object.object_id == "pose_landmarks":
    #                     number = MediapipeModelInfo.num_tracked_points_body
    #                 elif recorded_object.object_id == "face_landmarks":
    #                     number = MediapipeModelInfo.num_tracked_points_face
    #                 elif recorded_object.object_id == "left_hand_landmarks":
    #                     number = MediapipeModelInfo.num_tracked_points_left_hand
    #                 else:
    #                     number = MediapipeModelInfo.num_tracked_points_right_hand
    #                 for _ in range(number):
    #                     self.recorded_objects_array[i, landmark_number, :] = np.NaN
    #                     landmark_number += 1

    #     return self.recorded_objects_array
    
if __name__ == "__main__":
    array_1 = np.zeros((4, 3))
    array_2 = np.full((3, 3), np.nan)

    output = np.concatenate((array_1, array_2), axis=0)

    print(output)

    print(output.shape)

    print(output[5, :])
