from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation
from skellytracker.trackers.base_tracker.point_cloud import PointCloud
from skellytracker.trackers.mediapipe_tracker.mediapipe_names import (
    NUM_POSE_LANDMARKS,
    POSE_LANDMARK_NAMES,
)

_POSE_NAMES: tuple[str, ...] = tuple(POSE_LANDMARK_NAMES)


@dataclass(slots=True)
class MediapipePoseObservation(BaseObservation):
    tracker_type: str = field(default="mediapipe_pose", init=False)
    frame_number: int = 0
    image_size: tuple[int, int] = (0, 0)  # (height, width)

    points: PointCloud = field(default_factory=lambda: PointCloud.empty(_POSE_NAMES))

    # Extra data not in the PointCloud
    body_world_landmarks_xyz: NDArray = field(default_factory=lambda: np.full((NUM_POSE_LANDMARKS, 3), np.nan))
    segmentation_mask: np.ndarray | None = None

    @classmethod
    def from_detection_results(
        cls,
        frame_number: int,
        pose_landmarker_result: "mp.tasks.vision.PoseLandmarkerResult",
        image_size: tuple[int, int],
    ) -> "MediapipePoseObservation":
        """Convert a PoseLandmarkerResult into a MediapipePoseObservation."""
        height, width = image_size

        if len(pose_landmarker_result.pose_landmarks) == 0:
            return cls(frame_number=frame_number, image_size=image_size)

        landmarks = pose_landmarker_result.pose_landmarks[0]
        world_landmarks = pose_landmarker_result.pose_world_landmarks[0]

        body_xyz = np.array(
            [(lm.x * width, lm.y * height, lm.z * width) for lm in landmarks]
        )
        body_world_xyz = np.array(
            [(lm.x, lm.y, lm.z) for lm in world_landmarks]
        )
        visibility = np.array(
            [lm.visibility if lm.visibility is not None else 0.0 for lm in landmarks]
        )

        # Segmentation mask — squeeze to 2D if the Tasks API returns (H, W, 1)
        seg_mask = None
        if pose_landmarker_result.segmentation_masks:
            raw_mask = pose_landmarker_result.segmentation_masks[0].numpy_view().copy()
            seg_mask = raw_mask.squeeze()

        cloud = PointCloud(names=_POSE_NAMES, xyz=body_xyz, visibility=visibility)

        return cls(
            frame_number=frame_number,
            image_size=image_size,
            points=cloud,
            body_world_landmarks_xyz=body_world_xyz,
            segmentation_mask=seg_mask,
        )

    @property
    def has_detection(self) -> bool:
        """True if a body was detected (not all NaN)."""
        return self.points.n_valid > 0

    @property
    def body_landmarks_xyz(self) -> NDArray:
        return self.points.xyz

    @property
    def body_visibility(self) -> NDArray:
        return self.points.visibility
