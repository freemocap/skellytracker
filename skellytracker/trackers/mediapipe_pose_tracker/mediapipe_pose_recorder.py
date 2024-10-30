from copy import deepcopy
from typing import Dict
import numpy as np

from skellytracker.trackers.base_tracker.base_recorder import BaseRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.mediapipe_pose_tracker.mediapipe_pose_model_info import (
    MediapipePoseModelInfo,
)


class MediapipePoseRecorder(BaseRecorder):
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        self.recorded_objects.append(
            [
                deepcopy(tracked_objects[tracked_object_name])
                for tracked_object_name in MediapipePoseModelInfo.tracked_object_names
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
                MediapipePoseModelInfo.num_tracked_points,
                3,
            )
        )

        for i, recorded_object_list in enumerate(self.recorded_objects):
            recorded_object = recorded_object_list[0]
            if recorded_object.extra["landmarks"] is None:
                frame_data = np.full((MediapipePoseModelInfo.num_tracked_points, 3), np.nan)
            else:
                frame_data = np.array(
                    [
                        (landmark.x, landmark.y, landmark.z)
                        for landmark in recorded_object.extra["landmarks"].landmark
                    ]
                )
            self.recorded_objects_array[i] = frame_data

        # change from normalized image coordinates to pixel coordinates
        self.recorded_objects_array *= np.array([image_size[0], image_size[1], image_size[0]])  # multiply z by image width per mediapipe docs

        return self.recorded_objects_array

