from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.mediapipe_pose_tracker.mediapipe_pose_model_info import (
    MediapipeLandmarker,
    MediapipePoseModelInfo,
)
from skellytracker.trackers.mediapipe_pose_tracker.mediapipe_pose_recorder import (
    MediapipePoseRecorder,
)


class MediapipePoseTracker(BaseTracker):
    def __init__(
        self,
        model: MediapipeLandmarker = MediapipePoseModelInfo.heavy_model,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: VisionTaskRunningMode = VisionTaskRunningMode.IMAGE,  # TODO: Implement VIDEO (requires smoofing timestamps in some cases?), and maybe even LIVESTREAM for realtime?
        num_poses=1,
    ):
        super().__init__(
            tracked_object_names=MediapipePoseModelInfo.tracked_object_names,
            recorder=MediapipePoseRecorder(),
        )
        self.model = model
        model_path = self.model.download_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            output_segmentation_masks=False,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_poses=num_poses,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        # Convert the image to RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Process the image
        results = self.detector.detect(mp_image)

        # Update the tracking data
        self.tracked_objects["pose_landmarks"].extra[
            "landmarks"
        ] = results.pose_landmarks

        self.annotated_image = self.annotate_image(
            image=image, tracked_objects=self.tracked_objects
        )

        return self.tracked_objects

    def annotate_image(
        self, image: np.ndarray, tracked_objects: Dict[str, TrackedObject], **kwargs
    ) -> np.ndarray:
        annotated_image = image.copy()
        pose_landmarks_list = tracked_objects["pose_landmarks"].extra["landmarks"]

        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in pose_landmarks
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
        return annotated_image


if __name__ == "__main__":
    MediapipePoseTracker().demo()
