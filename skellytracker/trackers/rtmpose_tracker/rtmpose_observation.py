from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

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

    # Raw multi-person arrays as returned directly by the RTMPose model.
    # Shape: keypoints (num_persons, num_keypoints, 2), scores (num_persons, num_keypoints).
    # rtmlib returns keypoints as float64 and scores as float32 at runtime.
    keypoints: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0, 0), dtype=np.float64))
    scores: NDArray[np.float32] = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))

    @classmethod
    def from_detection_results(
            cls,
            frame_number: int,
            keypoints: NDArray[np.float64],
            scores: NDArray[np.float32],
            image_size: tuple[int, int],
    ) -> "RTMPoseObservation":
        # Take the first detected person
        if keypoints.shape[0] > 0:
            points_2d: NDArray[np.float64] = keypoints[0, :, :2].astype(np.float64)
            confidence: NDArray[np.float64] = scores[0, :].astype(np.float64)
        else:
            n = len(_RTMPOSE_NAMES)
            points_2d = np.full((n, 2), np.nan, dtype=np.float64)
            confidence = np.zeros(n, dtype=np.float64)

        n = points_2d.shape[0]
        xyz: NDArray[np.float64] = np.column_stack([points_2d, np.zeros(n, dtype=np.float64)])
        cloud = PointCloud(names=_RTMPOSE_NAMES, xyz=xyz, visibility=confidence)

        return cls(
            frame_number=frame_number,
            image_size=image_size,
            points=cloud,
            keypoints=keypoints,
            scores=scores,
        )
