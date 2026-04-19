from dataclasses import dataclass

import cv2
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseImageAnnotator,
    BaseImageAnnotatorConfig,
    BaseObservation,
)
from skellytracker.trackers.mediapipe_tracker.face.mediapipe_face_observation import MediapipeFaceObservation
from skellytracker.trackers.mediapipe_tracker.mediapipe_names import (
    FACEMESH_CONTOURS,
    FACEMESH_LEFT_IRIS,
    FACEMESH_RIGHT_IRIS,
)


class MediapipeFaceAnnotatorConfig(BaseImageAnnotatorConfig):
    show_overlay: bool = False
    contour_color: tuple[int, int, int] = (200, 244, 151)
    contour_thickness: int = 1
    left_iris_color: tuple[int, int, int] = (255, 2, 11)
    right_iris_color: tuple[int, int, int] = (2, 2, 211)
    iris_thickness: int = 2

@dataclass
class MediapipeFaceAnnotator(BaseImageAnnotator):
    config: MediapipeFaceAnnotatorConfig
    observations: list[MediapipeFaceObservation]

    @classmethod
    def create(cls, config: MediapipeFaceAnnotatorConfig) -> "MediapipeFaceAnnotator":
        return cls(config=config, observations=[])

    def annotate_image(self, image: np.ndarray, observation: BaseObservation) -> np.ndarray:
        if not isinstance(observation, MediapipeFaceObservation):
            raise TypeError(f"Expected MediapipeFaceObservation, got {type(observation)}")

        if not observation.has_detection:
            return image.copy()

        annotated = image.copy()
        all_points = observation.face_landmarks_xyz[:, :2]

        # Draw face contours (excluding iris — those get separate colors)
        self._draw_connections(
            image=annotated,
            points=all_points,
            connections=FACEMESH_CONTOURS,
            color=self.config.contour_color,
            thickness=self.config.contour_thickness,
        )

        # Draw irises with distinct colors
        self._draw_connections(
            image=annotated,
            points=all_points,
            connections=FACEMESH_LEFT_IRIS,
            color=self.config.left_iris_color,
            thickness=self.config.iris_thickness,
        )
        self._draw_connections(
            image=annotated,
            points=all_points,
            connections=FACEMESH_RIGHT_IRIS,
            color=self.config.right_iris_color,
            thickness=self.config.iris_thickness,
        )

        return annotated

    def _draw_connections(
        self,
        image: np.ndarray,
        points: np.ndarray,
        connections: list[tuple[int, int]],
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        for start_idx, end_idx in connections:
            if start_idx >= points.shape[0] or end_idx >= points.shape[0]:
                continue
            p1 = points[start_idx]
            p2 = points[end_idx]
            if np.isnan(p1).any() or np.isnan(p2).any():
                continue
            cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color=color, thickness=thickness)
