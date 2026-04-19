from dataclasses import dataclass

import cv2
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseImageAnnotator,
    BaseImageAnnotatorConfig,
    BaseObservation,
)
from skellytracker.trackers.mediapipe_tracker.body.mediapipe_pose_observation import MediapipePoseObservation
from skellytracker.trackers.mediapipe_tracker.mediapipe_names import POSE_CONNECTIONS

class MediapipePoseAnnotatorConfig(BaseImageAnnotatorConfig):
    show_overlay: bool = True
    landmark_color: tuple[int, int, int] = (245, 117, 66)
    landmark_radius: int = 4
    connection_color: tuple[int, int, int] = (245, 166, 230)
    connection_thickness: int = 2

@dataclass
class MediapipePoseAnnotator(BaseImageAnnotator):
    config: MediapipePoseAnnotatorConfig
    observations: list[MediapipePoseObservation]

    @classmethod
    def create(cls, config: MediapipePoseAnnotatorConfig) -> "MediapipePoseAnnotator":
        return cls(config=config, observations=[])

    def annotate_image(self, image: np.ndarray, observation: BaseObservation) -> np.ndarray:
        if not isinstance(observation, MediapipePoseObservation):
            raise TypeError(f"Expected MediapipePoseObservation, got {type(observation)}")

        if not observation.has_detection:
            return image.copy()

        annotated = image.copy()

        # Draw segmentation overlay
        if self.config.show_overlay and observation.segmentation_mask is not None:
            overlay = annotated.copy()
            mask = (observation.segmentation_mask * 50).astype("uint8")
            overlay[:, :, 2] = np.clip(overlay[:, :, 2].astype(np.int16) + mask, 0, 255).astype("uint8")
            annotated = overlay

        points = observation.body_landmarks_xyz[:, :2]

        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            p1 = points[start_idx]
            p2 = points[end_idx]
            if np.isnan(p1).any() or np.isnan(p2).any():
                continue
            cv2.line(
                annotated,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                color=self.config.connection_color,
                thickness=self.config.connection_thickness,
            )

        # Draw landmarks
        for i in range(points.shape[0]):
            pt = points[i]
            if np.isnan(pt).any():
                continue
            cv2.circle(
                annotated,
                (int(pt[0]), int(pt[1])),
                radius=self.config.landmark_radius,
                color=self.config.landmark_color,
                thickness=-1,
            )

        return annotated
