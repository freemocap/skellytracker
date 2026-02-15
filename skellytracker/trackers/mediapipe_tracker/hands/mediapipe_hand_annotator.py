import cv2
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseImageAnnotator,
    BaseImageAnnotatorConfig,
    BaseObservation,
)
from skellytracker.trackers.mediapipe_tracker.hands.mediapipe_hand_observation import MediapipeHandObservation
from skellytracker.trackers.mediapipe_tracker.mediapipe_names import HAND_CONNECTIONS


class MediapipeHandAnnotatorConfig(BaseImageAnnotatorConfig):
    show_overlay: bool = False
    right_hand_color: tuple[int, int, int] = (10, 22, 210)
    left_hand_color: tuple[int, int, int] = (230, 22, 20)
    connection_thickness: int = 2
    landmark_radius: int = 2


class MediapipeHandAnnotator(BaseImageAnnotator):
    config: MediapipeHandAnnotatorConfig
    observations: list[MediapipeHandObservation]

    @classmethod
    def create(cls, config: MediapipeHandAnnotatorConfig) -> "MediapipeHandAnnotator":
        return cls(config=config, observations=[])

    def annotate_image(self, image: np.ndarray, observation: BaseObservation) -> np.ndarray:
        if not isinstance(observation, MediapipeHandObservation):
            raise TypeError(f"Expected MediapipeHandObservation, got {type(observation)}")

        annotated = image.copy()

        if observation.has_right_hand:
            self._draw_hand(
                image=annotated,
                landmarks_xyz=observation.right_hand_landmarks_xyz,
                color=self.config.right_hand_color,
            )
        if observation.has_left_hand:
            self._draw_hand(
                image=annotated,
                landmarks_xyz=observation.left_hand_landmarks_xyz,
                color=self.config.left_hand_color,
            )

        return annotated

    def _draw_hand(
        self,
        image: np.ndarray,
        landmarks_xyz: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        points = landmarks_xyz[:, :2]

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            p1 = points[start_idx]
            p2 = points[end_idx]
            if np.isnan(p1).any() or np.isnan(p2).any():
                continue
            cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color=color, thickness=self.config.connection_thickness)

        # Draw landmarks
        for i in range(points.shape[0]):
            pt = points[i]
            if np.isnan(pt).any():
                continue
            cv2.circle(image, (int(pt[0]), int(pt[1])), radius=self.config.landmark_radius, color=color, thickness=-1)
