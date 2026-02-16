from dataclasses import dataclass, field

import numpy as np

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation
from skellytracker.trackers.base_tracker.point_cloud import PointCloud
from skellytracker.trackers.rtmpose_tracker.rtmpose_landmark_names import ALL_LANDMARK_NAMES

# VITPose uses COCO-WholeBody 133 keypoints — same topology as RTMPose
_VITPOSE_NAMES: tuple[str, ...] = tuple(ALL_LANDMARK_NAMES)
_NUM_KEYPOINTS = 133


@dataclass(slots=True)
class VITPoseObservation(BaseObservation):
    tracker_type: str = field(default="vitpose", init=False)
    frame_number: int = 0
    image_size: tuple[int, int] = (0, 0)
    points: PointCloud = field(default_factory=lambda: PointCloud.empty(_VITPOSE_NAMES))

    # Raw keypoints retained for access to confidence in original format
    raw_keypoints: np.ndarray = field(default_factory=lambda: np.full((_NUM_KEYPOINTS, 3), np.nan, dtype=np.float32))

    @classmethod
    def from_detection_results(
        cls,
        frame_number: int,
        results: dict[str, np.ndarray],
        image_size: tuple[int, int],
    ) -> "VITPoseObservation":
        if len(results) == 0:
            return cls(frame_number=frame_number, image_size=image_size)

        keypoints = results[0]  # First person only, shape (N, 3): x, y, confidence

        # VITPose returns (y, x) — swap to (x, y)
        points_xy = keypoints[:, :2][:, [1, 0]]
        confidence = keypoints[:, 2]

        n = points_xy.shape[0]
        xyz = np.column_stack([points_xy, np.zeros(n)])
        cloud = PointCloud(names=_VITPOSE_NAMES, xyz=xyz, visibility=confidence)

        return cls(
            frame_number=frame_number,
            image_size=image_size,
            points=cloud,
            raw_keypoints=keypoints,
        )
