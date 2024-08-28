import cv2
import mediapipe as mp
import numpy as np
from typing import Dict

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.mediapipe_tracker.mediapipe_holistic_recorder import (
    MediapipeHolisticRecorder,
)
from skellytracker.trackers.mediapipe_tracker.mediapipe_model_info import (
    MediapipeModelInfo,
)


class MediapipeHolisticTracker(BaseTracker):
    def __init__(
        self,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        smooth_landmarks=True,
    ):
        super().__init__(
            tracked_object_names=MediapipeModelInfo.tracked_object_names,
            recorder=MediapipeHolisticRecorder(),
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
            smooth_landmarks=smooth_landmarks,
        )

    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        # Convert the image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = self.holistic.process(rgb_image)

        # Update the tracking data
        self.tracked_objects["pose_landmarks"].extra[
            "landmarks"
        ] = results.pose_landmarks
        self.tracked_objects["face_landmarks"].extra[
            "landmarks"
        ] = results.face_landmarks
        self.tracked_objects["left_hand_landmarks"].extra[
            "landmarks"
        ] = results.left_hand_landmarks
        self.tracked_objects["right_hand_landmarks"].extra[
            "landmarks"
        ] = results.right_hand_landmarks

        self.annotated_image = self.annotate_image(
            image=image, tracked_objects=self.tracked_objects
        )

        return self.tracked_objects

    def annotate_image(
        self, image: np.ndarray, tracked_objects: Dict[str, TrackedObject], **kwargs
    ) -> np.ndarray:
        annotated_image = image.copy()
        # Draw the pose, face, and hand landmarks on the image
        self.mp_drawing.draw_landmarks(
            annotated_image,
            tracked_objects["pose_landmarks"].extra["landmarks"],
            self.mp_holistic.POSE_CONNECTIONS,
        )
        self.mp_drawing.draw_landmarks(
            annotated_image,
            tracked_objects["face_landmarks"].extra["landmarks"],
            self.mp_holistic.FACEMESH_TESSELATION,
        )
        self.mp_drawing.draw_landmarks(
            annotated_image,
            tracked_objects["left_hand_landmarks"].extra["landmarks"],
            self.mp_holistic.HAND_CONNECTIONS,
        )
        self.mp_drawing.draw_landmarks(
            annotated_image,
            tracked_objects["right_hand_landmarks"].extra["landmarks"],
            self.mp_holistic.HAND_CONNECTIONS,
        )

        return annotated_image


if __name__ == "__main__":
    MediapipeHolisticTracker(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        smooth_landmarks=True,
    ).demo()
