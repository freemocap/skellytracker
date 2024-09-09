import cv2
import numpy as np
import copy
import mediapipe as mp
import torch
from typing import Dict, Literal, Tuple
from ultralytics import YOLO

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.mediapipe_tracker.mediapipe_holistic_recorder import (
    MediapipeHolisticRecorder,
)
from skellytracker.trackers.mediapipe_tracker.mediapipe_model_info import (
    MediapipeModelInfo,
)
from skellytracker.trackers.yolo_object_tracker.yolo_object_model_info import (
    yolo_object_model_dictionary,
)


class YOLOMediapipeComboTracker(BaseTracker):
    def __init__(
        self,
        model_size: str = "nano",
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        smooth_landmarks: bool = True,
        bounding_box_buffer_percentage: float = 10,
        buffer_size_method: Literal[
            "buffer_by_box_size", "buffer_by_image_size"
        ] = "buffer_by_box_size",
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

        pytorch_model = yolo_object_model_dictionary[model_size]
        self.model = YOLO(pytorch_model)
        self.bounding_box_buffer_percentage = bounding_box_buffer_percentage
        self.buffer_size_method = buffer_size_method

    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:

        yolo_results = self.model(image, classes=0, max_det=1, verbose=False)
        box_xyxy = np.asarray(yolo_results[0].boxes.xyxy.cpu()).flatten()

        if box_xyxy.size > 0:
            box_left, box_top, box_right, box_bottom = box_xyxy

            if self.buffer_size_method == "buffer_by_image_size":
                width_buffer, height_buffer = self._get_buffer_bounding_box_total_image(
                    image, self.bounding_box_buffer_percentage
                )
            elif self.buffer_size_method == "buffer_by_box_size":
                width_buffer, height_buffer = self._get_buffer_bounding_box_box_size(
                    box_xyxy, self.bounding_box_buffer_percentage
                )
            else:
                raise ValueError(
                    f"Unknown buffer_size_method: {self.buffer_size_method}"
                )

            # Apply buffer, but set to original picture dimension if it goes out of bounds
            box_left = max(int(box_left - width_buffer), 0)
            box_top = max(int(box_top - height_buffer), 0)
            box_right = min(int(box_right + width_buffer), image.shape[1])
            box_bottom = min(int(box_bottom + height_buffer), image.shape[0])

            cropped_image = image[
                int(box_top) : int(box_bottom),
                int(box_left) : int(box_right),
            ]

            buffered_yolo_results = copy.deepcopy(yolo_results)
            buffered_yolo_results[0].boxes.xyxy[0] = torch.tensor(
                [box_left, box_top, box_right, box_bottom]
            )

        else:
            # eventually we should not even run mediapipe if no bbox is found
            box_left, box_top, box_right, box_bottom = 0, 0, 0, 0
            cropped_image = image

            buffered_yolo_results = yolo_results

        cropped_rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        mediapipe_results = self.holistic.process(cropped_rgb_image)

        # Update the tracking data
        self._rescale_cropped_data(
            image, box_left, box_top, box_right, box_bottom, mediapipe_results
        )
        self.tracked_objects["pose_landmarks"].extra[
            "landmarks"
        ] = mediapipe_results.pose_landmarks
        self.tracked_objects["face_landmarks"].extra[
            "landmarks"
        ] = mediapipe_results.face_landmarks
        self.tracked_objects["left_hand_landmarks"].extra[
            "landmarks"
        ] = mediapipe_results.left_hand_landmarks
        self.tracked_objects["right_hand_landmarks"].extra[
            "landmarks"
        ] = mediapipe_results.right_hand_landmarks

        bbox_image = buffered_yolo_results[0].plot()

        self.annotated_image = self.annotate_image(
            image=bbox_image, tracked_objects=self.tracked_objects
        )

        return self.tracked_objects

    def _get_buffer_bounding_box_total_image(
        self, image: np.ndarray, buffer_percentage: float
    ) -> Tuple[float, float]:
        width_buffer = image.shape[1] * (buffer_percentage / 100.0)
        height_buffer = image.shape[0] * (buffer_percentage / 100.0)

        return width_buffer, height_buffer

    def _get_buffer_bounding_box_box_size(
        self, box_xyxy: np.ndarray, buffer_percentage: float
    ) -> Tuple[float, float]:
        box_left, box_top, box_right, box_bottom = box_xyxy
        width_buffer = (box_right - box_left) * (buffer_percentage / 100.0)
        height_buffer = (box_bottom - box_top) * (buffer_percentage / 100.0)

        return width_buffer, height_buffer

    def _rescale_cropped_data(
        self,
        image: np.ndarray,
        box_left: int,
        box_top: int,
        box_right: int,
        box_bottom: int,
        mediapipe_results,
    ) -> None:
        if mediapipe_results.pose_landmarks is not None:
            for landmark in mediapipe_results.pose_landmarks.landmark:
                landmark.x = (
                    (landmark.x * (box_right - box_left)) + box_left
                ) / image.shape[1]
                landmark.y = (
                    (landmark.y * (box_bottom - box_top)) + box_top
                ) / image.shape[0]
                landmark.z = (landmark.z * (box_right - box_left)) / image.shape[1]
        if mediapipe_results.face_landmarks is not None:
            for landmark in mediapipe_results.face_landmarks.landmark:
                landmark.x = (
                    (landmark.x * (box_right - box_left)) + box_left
                ) / image.shape[1]
                landmark.y = (
                    (landmark.y * (box_bottom - box_top)) + box_top
                ) / image.shape[0]
                landmark.z = (landmark.z * (box_right - box_left)) / image.shape[1]
        if mediapipe_results.left_hand_landmarks is not None:
            for landmark in mediapipe_results.left_hand_landmarks.landmark:
                landmark.x = (
                    (landmark.x * (box_right - box_left)) + box_left
                ) / image.shape[1]
                landmark.y = (
                    (landmark.y * (box_bottom - box_top)) + box_top
                ) / image.shape[0]
                landmark.z = (landmark.z * (box_right - box_left)) / image.shape[1]
        if mediapipe_results.right_hand_landmarks is not None:
            for landmark in mediapipe_results.right_hand_landmarks.landmark:
                landmark.x = (
                    (landmark.x * (box_right - box_left)) + box_left
                ) / image.shape[1]
                landmark.y = (
                    (landmark.y * (box_bottom - box_top)) + box_top
                ) / image.shape[0]
                landmark.z = (landmark.z * (box_right - box_left)) / image.shape[1]

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
