from dataclasses import dataclass

import numpy as np
import torch

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation, TrackerType
from skellytracker.trackers.base_tracker.point_cloud import PointCloud
from skellytracker.trackers.rt_pose_tracker.names_and_connections import COCO_17_NAMES, RT_POSE_DEFINITION

_N_KEYPOINTS = 17


@dataclass
class RtPoseObservation(BaseObservation):
    frame_number: int
    image_size: tuple[int, int]
    points: PointCloud
    tracker_type: TrackerType = TrackerType.RT_POSE

    @classmethod
    def from_detection_results(
        cls,
        frame_number: int,
        keypoints_xy: torch.Tensor,
        scores: torch.Tensor,
        image_size: tuple[int, int],
    ) -> "RtPoseObservation":
        """
        Build an observation from the first detected person's keypoints.

        Args:
            frame_number: Index of the current frame.
            keypoints_xy: Tensor of shape (N, 17, 2) — x,y per person.
            scores: Tensor of shape (N, 17) — confidence per keypoint.
            image_size: (width, height) of the source image.
        """
        if keypoints_xy.shape[0] == 0:
            return cls(
                frame_number=frame_number,
                image_size=image_size,
                points=RT_POSE_DEFINITION.empty_point_cloud(),
            )

        kp = keypoints_xy[0].cpu().double().numpy()  # (17, 2) — float64 required by PointCloud
        vis = scores[0].cpu().double().numpy()       # (17,)  — float64 required by PointCloud
        xyz = np.column_stack([kp, np.zeros(_N_KEYPOINTS)])

        return cls(
            frame_number=frame_number,
            image_size=image_size,
            points=PointCloud(names=COCO_17_NAMES, xyz=xyz, visibility=vis),
        )
