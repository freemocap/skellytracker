import cv2
import numpy as np
import mediapipe as mp
from typing import Dict
from ultralytics import YOLO

from skelly_tracker.trackers.base_tracker.base_tracker import BaseTracker
from skelly_tracker.trackers.base_tracker.tracked_object import TrackedObject
from skelly_tracker.trackers.mediapipe_tracker.mediapipe_holistic_recorder import (
    MediapipeHolisticRecorder,
)
from skelly_tracker.trackers.mediapipe_tracker.mediapipe_model_info import (
    MediapipeModelInfo,
)
from skelly_tracker.trackers.yolo_object_tracker.yolo_object_model_dictionary import (
    yolo_object_model_dictionary,
)


class YOLOMediapipeComboTracker(BaseTracker):
    def __init__(
        self,
        model_size: str = "nano",
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        smooth_landmarks=True,
    ):
        super().__init__(
            tracked_object_names=MediapipeModelInfo.mediapipe_tracked_object_names,
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

        pytorch_model = yolo_object_model_dictionary[model_size]
        self.model = YOLO(pytorch_model)

    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        yolo_results = self.model(image, classes=0, max_det=1, verbose=False)

        box_xywh = np.asarray(yolo_results[0].boxes.xywh).flatten()

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if box_xywh.size > 0:
            box_x, box_y, box_width, box_height = box_xywh
            cropped_rgb_image = rgb_image[
                int(box_y) : int(box_y + box_height),
                int(box_x) : int(box_x + box_width),
            ]
        else:
            cropped_rgb_image = rgb_image

        mediapipe_results = self.holistic.process(cropped_rgb_image)

        # Update the tracking data
        if mediapipe_results.pose_landmarks is not None:
            for landmark in mediapipe_results.pose_landmarks.landmark:
                print(landmark.y)
                landmark.x = ((landmark.x * box_width) + box_x) / image.shape[1]
                landmark.y = ((landmark.y * box_height) + box_y) / image.shape[0]
                landmark.z = (landmark.z * box_height) / image.shape[0]
                print(landmark.y)
        self.tracked_objects["pose_landmarks"].extra[
            "landmarks"
        ] = mediapipe_results.pose_landmarks
        if mediapipe_results.face_landmarks is not None:
            for landmark in mediapipe_results.face_landmarks.landmark:
                landmark.x = ((landmark.x * box_width) + box_x) / image.shape[1]
                landmark.y = ((landmark.y * box_height) + box_y) / image.shape[0]
                landmark.z = ((landmark.z) * box_height) / image.shape[0]
        self.tracked_objects["face_landmarks"].extra[
            "landmarks"
        ] = mediapipe_results.face_landmarks
        if mediapipe_results.left_hand_landmarks is not None:
            for landmark in mediapipe_results.left_hand_landmarks.landmark:
                landmark.x = ((landmark.x * box_width) + box_x) / image.shape[1]
                landmark.y = ((landmark.y * box_height) + box_y) / image.shape[0]
                landmark.z = (landmark.z * box_height) / image.shape[0]
        self.tracked_objects["left_hand_landmarks"].extra[
            "landmarks"
        ] = mediapipe_results.left_hand_landmarks
        if mediapipe_results.right_hand_landmarks is not None:
            for landmark in mediapipe_results.right_hand_landmarks.landmark:
                landmark.x = ((landmark.x * box_width) + box_x) / image.shape[1]
                landmark.y = ((landmark.y * box_height) + box_y) / image.shape[0]
                landmark.z = (landmark.z * box_height) / image.shape[0]
        self.tracked_objects["right_hand_landmarks"].extra[
            "landmarks"
        ] = mediapipe_results.right_hand_landmarks

        self.raw_image = image.copy()

        self.annotated_image = self.annotate_image(
            image=image, tracked_objects=self.tracked_objects
        )

        return self.tracked_objects

    def annotate_image(
        self, image: np.ndarray, tracked_objects: Dict[str, TrackedObject], **kwargs
    ) -> np.ndarray:
        # Draw the pose, face, and hand landmarks on the image
        self.mp_drawing.draw_landmarks(
            image,
            tracked_objects["pose_landmarks"].extra["landmarks"],
            self.mp_holistic.POSE_CONNECTIONS,
        )
        self.mp_drawing.draw_landmarks(
            image,
            tracked_objects["face_landmarks"].extra["landmarks"],
            self.mp_holistic.FACEMESH_TESSELATION,
        )
        self.mp_drawing.draw_landmarks(
            image,
            tracked_objects["left_hand_landmarks"].extra["landmarks"],
            self.mp_holistic.HAND_CONNECTIONS,
        )
        self.mp_drawing.draw_landmarks(
            image,
            tracked_objects["right_hand_landmarks"].extra["landmarks"],
            self.mp_holistic.HAND_CONNECTIONS,
        )

        return image


if __name__ == "__main__":
    YOLOMediapipeComboTracker().demo()
