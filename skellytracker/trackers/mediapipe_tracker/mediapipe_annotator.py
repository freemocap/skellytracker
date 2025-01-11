from dataclasses import field, dataclass

import numpy as np
import cv2
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import drawing_utils

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


@dataclass
class MediapipeImageAnnotator(BaseImageAnnotator):
    config: MediapipeAnnotatorConfig
    observations: list[MediapipeObservation] = field(default_factory=list)

    @classmethod
    def create(cls, config: MediapipeAnnotatorConfig):
        return cls(config=config)

    def annotate_image(
            self,
            image: np.ndarray,
            latest_observation: MediapipeObservation | None = None,
    ) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        text_offset = int(image_height * 0.01)

        if latest_observation is None:
            return image.copy()
        # Copy the original image for annotation
        annotated_image = image.copy()

        drawing_utils.draw_landmarks(
            annotated_image,
            latest_observation.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
        )
        drawing_utils.draw_landmarks(
            annotated_image,
            latest_observation.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )
        drawing_utils.draw_landmarks(
            annotated_image,
            latest_observation.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )
        drawing_utils.draw_landmarks(
            annotated_image,
            latest_observation.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
        )

        return annotated_image
