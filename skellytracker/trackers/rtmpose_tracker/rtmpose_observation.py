from dataclasses import dataclass, field

import numpy as np

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation
from skellytracker.trackers.base_tracker.point_cloud import PointCloud
from skellytracker.trackers.rtmpose_tracker.rtmpose_landmark_names import ALL_LANDMARK_NAMES

_RTMPOSE_NAMES: tuple[str, ...] = tuple(ALL_LANDMARK_NAMES)


@dataclass(slots=True)
class RTMPoseObservation(BaseObservation):
    tracker_type: str = field(default="rtmpose", init=False)
    frame_number: int = 0
    image_size: tuple[int, int] = (0, 0)
    points: PointCloud = field(default_factory=lambda: PointCloud.empty(_RTMPOSE_NAMES))

    # Raw multi-person arrays retained for access to other detected persons
    keypoints: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 0)))
    scores: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    @classmethod
    def from_detection_results(
        cls,
        frame_number: int,
        keypoints: np.ndarray,
        scores: np.ndarray,
        image_size: tuple[int, int],
    ) -> "RTMPoseObservation":
        # Take the first detected person
        if keypoints.shape[0] > 0:
            points_2d = keypoints[0, :, :2]  # (N, 2)
            confidence = scores[0, :]         # (N,)
        else:
            n = len(_RTMPOSE_NAMES)
            points_2d = np.full((n, 2), np.nan)
            confidence = np.zeros(n)

        n = points_2d.shape[0]
        xyz = np.column_stack([points_2d, np.zeros(n)])
        cloud = PointCloud(names=_RTMPOSE_NAMES, xyz=xyz, visibility=confidence)

        return cls(
            frame_number=frame_number,
            image_size=image_size,
            points=cloud,
            keypoints=keypoints,
            scores=scores,
        )
