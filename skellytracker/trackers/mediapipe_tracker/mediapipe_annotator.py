import cv2
import numpy as np
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_IRIS, FACEMESH_LEFT_IRIS
from numpydantic import NDArray, Shape

from skellytracker.trackers.base_tracker.base_tracker import BaseImageAnnotatorConfig, BaseImageAnnotator
from skellytracker.trackers.mediapipe_tracker.mediapipe_observation import MediapipeObservation


class MediapipeAnnotatorConfig(BaseImageAnnotatorConfig):
    show_tracks: int | None = 15
    corner_marker_type: int = cv2.MARKER_DIAMOND
    corner_marker_size: int = 10
    corner_marker_thickness: int = 2
    corner_marker_color: tuple[int, int, int] = (0, 0, 255)

    aruco_lines_thickness: int = 2
    aruco_lines_color: tuple[int, int, int] = (0, 255, 0)

    text_color: tuple[int, int, int] = (215, 115, 40)
    text_size: float = .5
    text_thickness: int = 2
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX


class MediapipeImageAnnotator(BaseImageAnnotator):
    config: MediapipeAnnotatorConfig
    observations: list[MediapipeObservation]

    @classmethod
    def create(cls, config: MediapipeAnnotatorConfig):
        return cls(config=config, observations=[])

    def annotate_image(
            self,
            image: NDArray[Shape["* width, * height, 1-4 channels"], np.uint8],
            latest_observation: MediapipeObservation | None = None,
    ) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        text_offset = int(image_height * 0.01)

        if latest_observation is None:
            return image.copy()
        # Copy the original image for annotation
        annotated_image = image.copy()

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=latest_observation.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(245, 166, 230), thickness=2),
            is_drawing_landmarks=True,
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=latest_observation.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(color=(10, 22, 210), thickness=1, circle_radius=2),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(10, 44, 211), thickness=2),
            is_drawing_landmarks=True,
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=latest_observation.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(color=(230, 22, 20), thickness=1, circle_radius=2),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(230, 44, 20), thickness=2),
            is_drawing_landmarks=True,
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=latest_observation.face_landmarks,
            connections=mp_holistic.FACEMESH_CONTOURS,
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(280, 244, 151), thickness=2),
            is_drawing_landmarks=False
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=latest_observation.face_landmarks,
            connections=FACEMESH_RIGHT_IRIS,
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(2, 2, 211), thickness=2),
            is_drawing_landmarks=False
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=latest_observation.face_landmarks,
            connections=FACEMESH_LEFT_IRIS,
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(255, 2, 11), thickness=2),
            is_drawing_landmarks=False
        )
        return annotated_image
