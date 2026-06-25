from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation
from skellytracker.trackers.base_tracker.point_cloud import PointCloud
from skellytracker.trackers.rtmpose_tracker.names_and_connections import RTMPOSE_WHOLEBODY_DEFINITION

# Names and order come from the YAML-authored wholebody definition.
# The composition YAML lays points out in body → right_hand → left_hand → face order.
_RTMPOSE_NAMES: tuple[str, ...] = RTMPOSE_WHOLEBODY_DEFINITION.tracked_points

# rtmlib's Wholebody model returns 133 keypoints in COCO-WholeBody order:
#   body(0..22) + face(23..90) + left_hand(91..111) + right_hand(112..132)
# The wholebody YAML composes as:
#   body(0..22) + right_hand(23..43) + left_hand(44..64) + face(65..132)
# This permutation maps rtmlib source index → target (schema) index.
_RTMLIB_TO_SCHEMA_PERM: NDArray[np.intp] = np.concatenate([
    np.arange(0, 23, dtype=np.intp),      # body stays in place
    np.arange(112, 133, dtype=np.intp),   # right_hand moves up
    np.arange(91, 112, dtype=np.intp),    # left_hand moves up
    np.arange(23, 91, dtype=np.intp),     # face moves down
])
assert _RTMLIB_TO_SCHEMA_PERM.shape == (133,)
assert len(_RTMPOSE_NAMES) == 133


@dataclass(slots=True)
class RTMPoseObservation(BaseObservation):
    tracker_type: str = field(default="rtmpose", init=False)
    frame_number: int = 0
    image_size: tuple[int, int] = (0, 0)
    points: PointCloud = field(default_factory=lambda: PointCloud.empty(_RTMPOSE_NAMES))

    # Raw multi-person arrays as returned directly by the RTMPose model.
    # Shape: keypoints (num_persons, num_keypoints, 2), scores (num_persons, num_keypoints).
    # rtmlib returns keypoints as float64 and scores as float32 at runtime.
    # NOTE: these keep rtmlib's native index ordering, not the schema ordering.
    keypoints: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0, 0), dtype=np.float64))
    scores: NDArray[np.float32] = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))

    # Bbox that produced this observation (xyxy, image pixel coords).
    # None when not available. True → from YOLOX detector, False → tracking.
    bbox: NDArray | None = None
    bbox_from_detector: bool = True

    @classmethod
    def from_detection_results(
            cls,
            frame_number: int,
            keypoints: NDArray[np.float64],
            scores: NDArray[np.float32],
            image_size: tuple[int, int],
            bbox: NDArray | None = None,
            bbox_from_detector: bool = True,
    ) -> "RTMPoseObservation":
        # Take the first detected person
        if keypoints.shape[0] > 0:
            points_2d: NDArray[np.float64] = keypoints[0, :, :2].astype(np.float64)
            confidence: NDArray[np.float64] = scores[0, :].astype(np.float64)
        else:
            n = len(_RTMPOSE_NAMES)
            points_2d = np.full((n, 2), np.nan, dtype=np.float64)
            confidence = np.zeros(n, dtype=np.float64)

        # Permute rtmlib's native order into the schema composition order so the
        # i-th PointCloud row matches _RTMPOSE_NAMES[i].
        points_2d = points_2d[_RTMLIB_TO_SCHEMA_PERM]
        confidence = confidence[_RTMLIB_TO_SCHEMA_PERM]

        n = points_2d.shape[0]
        xyz: NDArray[np.float64] = np.column_stack([points_2d, np.zeros(n, dtype=np.float64)])
        cloud = PointCloud(names=_RTMPOSE_NAMES, xyz=xyz, visibility=confidence)

        return cls(
            frame_number=frame_number,
            image_size=image_size,
            points=cloud,
            keypoints=keypoints,
            scores=scores,
            bbox=bbox.astype(np.float64) if bbox is not None else None,
            bbox_from_detector=bbox_from_detector,
        )
