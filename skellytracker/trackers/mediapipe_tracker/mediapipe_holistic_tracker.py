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
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        # Initialize drawing styles
        self._initialize_drawing_styles()

    def _initialize_drawing_styles(self):
        # Pose
        self.pose_landmark_style = self.mp_drawing_styles.get_default_pose_landmarks_style()
        self.pose_connection_style = self.mp_drawing.DrawingSpec(
            color=(177, 171, 95),  # BGR convention
            thickness=2
        )
        
        # Iterate over all defined pose landmarks to set left-right styles
        for landmark_enum in self.mp_pose.PoseLandmark:
            self.pose_landmark_style[landmark_enum].circle_radius = 2
            self.pose_landmark_style[landmark_enum].thickness = 2
            name = landmark_enum.name.lower()
            if "left" in name:
                self.pose_landmark_style[landmark_enum].color = (255, 0, 0)  # blue BGR convention
            elif "right" in name:
                self.pose_landmark_style[landmark_enum].color = (0, 0, 255)  # red BGR convention
            else:
                self.pose_landmark_style[landmark_enum].color = (255, 255, 255)  # white for central

        # Hands
        self.hands_landmark_style = self.mp_drawing_styles.get_default_hand_landmarks_style()
        # Set circle radius of hand landmarks
        for spec in self.hands_landmark_style.values():
            spec.circle_radius = 3

        # Face
        self.face_mesh_tesselation_style = self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        self.face_mesh_tesselation_style.color = (255, 255, 255)
        self.face_mesh_tesselation_style.thickness = 0
        self.face_mesh_tesselation_style.circle_radius = 1

        self.face_mesh_contours_style = self.mp_drawing_styles.get_default_face_mesh_contours_style(1)

        return

    def process_image(
        self, image: np.ndarray, annotate_image: bool = True, **kwargs
    ) -> Dict[str, TrackedObject]:
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

        if annotate_image:
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
            landmark_drawing_spec=self.pose_landmark_style,
            connection_drawing_spec=self.pose_connection_style,
        )
        self.mp_drawing.draw_landmarks(
            annotated_image,
            tracked_objects["face_landmarks"].extra["landmarks"],
            self.mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=self.face_mesh_tesselation_style,
            connection_drawing_spec=self.face_mesh_contours_style,
        )
        self.mp_drawing.draw_landmarks(
            annotated_image,
            tracked_objects["left_hand_landmarks"].extra["landmarks"],
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.hands_landmark_style,
        )
        self.mp_drawing.draw_landmarks(
            annotated_image,
            tracked_objects["right_hand_landmarks"].extra["landmarks"],
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.hands_landmark_style,
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
