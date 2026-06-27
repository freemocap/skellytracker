from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from skellytracker.old.base_tracker.base_tracker_abcs import BaseObservation
from skellytracker.old.base_tracker.point_cloud import PointCloud
from skellytracker.old.composite_gpu_tracker.names_and_connections import (
    RTMO_HYBRID_DEFINITION,
)
from skellytracker.old.composite_gpu_tracker.roi_crop_utils import ROIBox

_HYBRID_NAMES: tuple[str, ...] = RTMO_HYBRID_DEFINITION.tracked_points

# Slice boundaries from the composition definition.
_BODY_START = 0
_BODY_END = 17  # RTMO body: 17 COCO keypoints
_RHAND_START = _BODY_END  # 17
_RHAND_END = _RHAND_START + 21  # 38
_LHAND_START = _RHAND_END  # 38
_LHAND_END = _LHAND_START + 21  # 59
_FACE_START = _LHAND_END  # 59
_FACE_END = _FACE_START + 106  # 165


@dataclass(slots=True)
class CompositeGPUObservation(BaseObservation):
    tracker_type: str = field(default="rtmo_hybrid", init=False)
    frame_number: int = 0
    image_size: tuple[int, int] = (0, 0)
    points: PointCloud = field(default_factory=lambda: PointCloud.empty(_HYBRID_NAMES))

    # Raw sub-model arrays, keyed by component name.
    # Body:  (num_persons, 17, 2) / (num_persons, 17)
    # Hands: (1, 42, 2) / (1, 42) — concatenated right(0:21)+left(21:42)
    # Face:  (1, 68, 2) / (1, 68)
    body_keypoints: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 17, 0), dtype=np.float64))
    body_scores: NDArray[np.float32] = field(default_factory=lambda: np.empty((0, 17), dtype=np.float32))
    hands_keypoints: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 42, 0), dtype=np.float64))
    hands_scores: NDArray[np.float32] = field(default_factory=lambda: np.empty((0, 42), dtype=np.float32))
    face_keypoints: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 106, 0), dtype=np.float64))
    face_scores: NDArray[np.float32] = field(default_factory=lambda: np.empty((0, 106), dtype=np.float32))

    # Pre-cleanup hand keypoints (before wrist blending + anthropometry filter).
    # Same shape as hands_keypoints: (1, 42, 2) / (1, 42).
    raw_hands_keypoints: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 42, 0), dtype=np.float64))
    raw_hands_scores: NDArray[np.float32] = field(default_factory=lambda: np.empty((0, 42), dtype=np.float32))

    # ROI crop boxes (for debug visualization)
    right_hand_roi: ROIBox | None = None
    left_hand_roi: ROIBox | None = None
    face_roi: ROIBox | None = None

    @classmethod
    def from_detection_results(
        cls,
        *,
        frame_number: int,
        image_size: tuple[int, int],
        body_keypoints: NDArray[np.float64],
        body_scores: NDArray[np.float32],
        hands_keypoints: NDArray[np.float64],
        hands_scores: NDArray[np.float32],
        face_keypoints: NDArray[np.float64],
        face_scores: NDArray[np.float32],
        right_hand_roi: ROIBox | None = None,
        left_hand_roi: ROIBox | None = None,
        face_roi: ROIBox | None = None,
        raw_hands_keypoints: NDArray[np.float64] | None = None,
        raw_hands_scores: NDArray[np.float32] | None = None,
    ) -> "CompositeGPUObservation":
        """
        Build a composite observation from raw sub-model outputs.

        Takes the first detected person from body and merges all components
        into a single PointCloud with schema-aligned names.
        """
        n = len(_HYBRID_NAMES)
        xyz = np.full((n, 3), np.nan, dtype=np.float64)
        visibility = np.zeros(n, dtype=np.float64)

        # Body: first detected person
        if body_keypoints.shape[0] > 0:
            body_xy = body_keypoints[0, :, :2].astype(np.float64)
            body_vis = body_scores[0, :].astype(np.float64)
            xyz[_BODY_START:_BODY_END, :2] = body_xy
            visibility[_BODY_START:_BODY_END] = body_vis

        # Hands: concatenated right(21) + left(21)
        if hands_keypoints.shape[0] > 0 and hands_keypoints.shape[1] >= 42:
            hands_xy = hands_keypoints[0, :42, :2].astype(np.float64)
            hands_vis = hands_scores[0, :42].astype(np.float64)
            xyz[_RHAND_START:_RHAND_END, :2] = hands_xy[:21]
            visibility[_RHAND_START:_RHAND_END] = hands_vis[:21]
            xyz[_LHAND_START:_LHAND_END, :2] = hands_xy[21:42]
            visibility[_LHAND_START:_LHAND_END] = hands_vis[21:42]

        # Face (106 LaPa keypoints)
        if face_keypoints.shape[0] > 0 and face_keypoints.shape[1] >= 106:
            face_xy = face_keypoints[0, :106, :2].astype(np.float64)
            face_vis = face_scores[0, :106].astype(np.float64)
            xyz[_FACE_START:_FACE_END, :2] = face_xy
            visibility[_FACE_START:_FACE_END] = face_vis

        cloud = PointCloud(names=_HYBRID_NAMES, xyz=xyz, visibility=visibility)

        return cls(
            frame_number=frame_number,
            image_size=image_size,
            points=cloud,
            body_keypoints=body_keypoints,
            body_scores=body_scores,
            hands_keypoints=hands_keypoints,
            hands_scores=hands_scores,
            face_keypoints=face_keypoints,
            face_scores=face_scores,
            right_hand_roi=right_hand_roi,
            left_hand_roi=left_hand_roi,
            face_roi=face_roi,
            raw_hands_keypoints=(
                raw_hands_keypoints
                if raw_hands_keypoints is not None
                else np.empty((0, 42, 0), dtype=np.float64)
            ),
            raw_hands_scores=(
                raw_hands_scores
                if raw_hands_scores is not None
                else np.empty((0, 42), dtype=np.float32)
            ),
        )
