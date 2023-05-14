import cv2
import mediapipe as mp
import numpy as np
from skelly_tracker.trackers.base_tracker.base_tracker import BaseTracker

class MediapipeHolisticTracker(BaseTracker):
    def __init__(self, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False, skip_2d_image_tracking=False):
        super().__init__(tracked_object_names=["pose_landmarks", "face_landmarks", "left_hand_landmarks", "right_hand_landmarks"])
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
            smooth_landmarks=not skip_2d_image_tracking
        )

    def process_image(self, image, **kwargs):
        # Convert the image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = self.holistic.process(rgb_image)

        # Draw the pose, face, and hand landmarks on the image
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION)
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        # Update the tracking data
        self.tracking_data = {
            "pose_landmarks": results.pose_landmarks,
            "face_landmarks": results.face_landmarks,
            "left_hand_landmarks": results.left_hand_landmarks,
            "right_hand_landmarks": results.right_hand_landmarks
        }
        self.annotated_image = image

        return {
            "tracking_data": self.tracking_data,
            "annotated_image": self.annotated_image,
            "raw_image": image,
        }

if __name__ == "__main__":
    MediapipeHolisticTracker().demo()
