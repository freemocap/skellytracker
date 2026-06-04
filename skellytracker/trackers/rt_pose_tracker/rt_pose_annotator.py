from dataclasses import dataclass

import cv2
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseImageAnnotator, BaseImageAnnotatorConfig
from skellytracker.trackers.rt_pose_tracker.names_and_connections import RT_POSE_DEFINITION
from skellytracker.trackers.rt_pose_tracker.rt_pose_observation import RtPoseObservation

_CONNECTION_INDICES = RT_POSE_DEFINITION.connection_indices()


@dataclass
class RtPoseAnnotatorConfig(BaseImageAnnotatorConfig):
    keypoint_color: tuple[int, int, int] = (0, 255, 0)
    connection_color: tuple[int, int, int] = (255, 165, 0)
    radius: int = 5
    thickness: int = 2
    confidence_threshold: float = 0.3


class RtPoseAnnotator(BaseImageAnnotator):
    config: RtPoseAnnotatorConfig
    observations: list[RtPoseObservation]

    @classmethod
    def create(cls, config: RtPoseAnnotatorConfig | None = None) -> "RtPoseAnnotator":
        return cls(config=config or RtPoseAnnotatorConfig(), observations=[])

    def annotate_image(self, image: np.ndarray, observation: RtPoseObservation) -> np.ndarray:
        annotated = image.copy()
        points = observation.points
        threshold = self.config.confidence_threshold

        for i, j in _CONNECTION_INDICES:
            if points.visibility[i] >= threshold and points.visibility[j] >= threshold:
                pt1 = (int(points.xyz[i, 0]), int(points.xyz[i, 1]))
                pt2 = (int(points.xyz[j, 0]), int(points.xyz[j, 1]))
                cv2.line(annotated, pt1, pt2, self.config.connection_color, self.config.thickness)

        for idx in range(points.n_points):
            if points.visibility[idx] >= threshold:
                cx = int(points.xyz[idx, 0])
                cy = int(points.xyz[idx, 1])
                cv2.circle(annotated, (cx, cy), self.config.radius, self.config.keypoint_color, -1)

        return annotated
