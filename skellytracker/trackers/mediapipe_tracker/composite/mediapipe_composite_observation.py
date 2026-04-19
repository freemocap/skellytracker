from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation
from skellytracker.trackers.base_tracker.point_cloud import PointCloud
from skellytracker.trackers.mediapipe_tracker.body.mediapipe_pose_observation import MediapipePoseObservation
from skellytracker.trackers.mediapipe_tracker.composite.composite_tracker_mappings import (
    FACE_TO_POSE_DIRECT_MAP,
    IRIS_TO_POSE_MAP,
    LEFT_HAND_TO_POSE_MAP,
    POSE_LEFT_WRIST_FUSE_WITH_HAND_WRIST,
    POSE_RIGHT_WRIST_FUSE_WITH_HAND_WRIST,
    RIGHT_HAND_TO_POSE_MAP,
)
from skellytracker.trackers.mediapipe_tracker.face.mediapipe_face_observation import (
    MediapipeFaceObservation,
    NUM_FACE_LANDMARKS,
)
from skellytracker.trackers.mediapipe_tracker.hands.mediapipe_hand_observation import (
    MediapipeHandObservation,
    NUM_HAND_LANDMARKS,
)
from skellytracker.trackers.mediapipe_tracker.names_and_connections import (
    MEDIAPIPE_BODY_DEFINITION,
    MEDIAPIPE_FACE_CONTOUR_DEFINITION,
    MEDIAPIPE_HOLISTIC_DEFINITION,
)

NUM_POSE_LANDMARKS: int = MEDIAPIPE_BODY_DEFINITION.num_tracked_points

# Slice boundaries for the concatenated holistic PointCloud, derived from
# the composition YAML so they stay in sync with MEDIAPIPE_HOLISTIC_DEFINITION.
_BODY_START = 0
_BODY_END = _BODY_START + NUM_POSE_LANDMARKS
_RHAND_START = _BODY_END
_RHAND_END = _RHAND_START + NUM_HAND_LANDMARKS
_LHAND_START = _RHAND_END
_LHAND_END = _LHAND_START + NUM_HAND_LANDMARKS
_FACE_START = _LHAND_END
_FACE_END = _FACE_START + MEDIAPIPE_FACE_CONTOUR_DEFINITION.num_tracked_points

_HOLISTIC_NAMES: tuple[str, ...] = MEDIAPIPE_HOLISTIC_DEFINITION.tracked_points
assert len(_HOLISTIC_NAMES) == _FACE_END, (
    f"Holistic composition size {_FACE_END} disagrees with YAML length {len(_HOLISTIC_NAMES)}"
)

# Row indices of the face contour subset within the tesselated 478-point array.
# Used to subset the face detector's native output when building the composite.
from skellytracker.trackers.mediapipe_tracker.names_and_connections import (  # noqa: E402
    MEDIAPIPE_FACE_TESSELATED_DEFINITION,
)

_FACE_CONTOUR_SUBSET_INDICES: tuple[int, ...] = tuple(
    MEDIAPIPE_FACE_TESSELATED_DEFINITION.index_of(name)
    for name in MEDIAPIPE_FACE_CONTOUR_DEFINITION.tracked_points
)


class ROIBox:
    """Bounding box for an ROI crop in full-image coordinates."""

    __slots__ = ("x", "y", "width", "height")

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass(slots=True)
class MediapipeCompositeObservation(BaseObservation):
    """
    Holistic-style observation combining pose, hand, and face detections.

    All landmark data is stored in a single PointCloud whose names/ordering
    come from the mediapipe_holistic.yaml composition definition.

    Body landmarks are FUSED at construction time: higher-precision
    hand/face data is spliced in where available.
    """

    tracker_type: str = field(default="mediapipe_composite", init=False)
    frame_number: int = 0
    image_size: tuple[int, int] = (0, 0)  # (height, width)

    points: PointCloud = field(default_factory=MEDIAPIPE_HOLISTIC_DEFINITION.empty_point_cloud)

    # Sub-observations retained for metadata (blendshapes, segmentation, world coords)
    pose: MediapipePoseObservation | None = None
    hands: MediapipeHandObservation | None = None
    face: MediapipeFaceObservation | None = None

    left_hand_roi: ROIBox | None = None
    right_hand_roi: ROIBox | None = None
    face_roi: ROIBox | None = None

    @classmethod
    def from_detection_results(cls, **kwargs: object) -> "MediapipeCompositeObservation":
        raise NotImplementedError("Use MediapipeCompositeObservation.build()")

    @classmethod
    def build(
        cls,
        frame_number: int,
        image_size: tuple[int, int],
        pose: MediapipePoseObservation | None,
        hands: MediapipeHandObservation | None,
        face: MediapipeFaceObservation | None,
        left_hand_roi: ROIBox | None = None,
        right_hand_roi: ROIBox | None = None,
        face_roi: ROIBox | None = None,
    ) -> "MediapipeCompositeObservation":
        """Build by concatenating sub-detections into a single fused PointCloud."""
        body_xyz = pose.body_landmarks_xyz.copy() if pose is not None and pose.has_detection else np.full((NUM_POSE_LANDMARKS, 3), np.nan)
        body_vis = pose.body_visibility.copy() if pose is not None else np.zeros(NUM_POSE_LANDMARKS)

        rh_xyz = hands.right_hand_landmarks_xyz.copy() if hands is not None and hands.has_right_hand else np.full((NUM_HAND_LANDMARKS, 3), np.nan)
        rh_vis = hands.right_hand_visibility.copy() if hands is not None else np.zeros(NUM_HAND_LANDMARKS)

        lh_xyz = hands.left_hand_landmarks_xyz.copy() if hands is not None and hands.has_left_hand else np.full((NUM_HAND_LANDMARKS, 3), np.nan)
        lh_vis = hands.left_hand_visibility.copy() if hands is not None else np.zeros(NUM_HAND_LANDMARKS)

        face_contour_xyz, face_contour_vis = cls._extract_face_contour(face)

        cls._fuse_body_with_face(body_xyz=body_xyz, face=face)
        cls._fuse_body_with_hand(body_xyz=body_xyz, hand_xyz=rh_xyz, hand_to_pose_map=RIGHT_HAND_TO_POSE_MAP, wrist_pair=POSE_RIGHT_WRIST_FUSE_WITH_HAND_WRIST)
        cls._fuse_body_with_hand(body_xyz=body_xyz, hand_xyz=lh_xyz, hand_to_pose_map=LEFT_HAND_TO_POSE_MAP, wrist_pair=POSE_LEFT_WRIST_FUSE_WITH_HAND_WRIST)

        xyz = np.concatenate([body_xyz, rh_xyz, lh_xyz, face_contour_xyz], axis=0)
        vis = np.concatenate([body_vis, rh_vis, lh_vis, face_contour_vis], axis=0)
        cloud = PointCloud(names=_HOLISTIC_NAMES, xyz=xyz, visibility=vis)

        return cls(
            frame_number=frame_number,
            image_size=image_size,
            points=cloud,
            pose=pose,
            hands=hands,
            face=face,
            left_hand_roi=left_hand_roi,
            right_hand_roi=right_hand_roi,
            face_roi=face_roi,
        )

    # =========================================================================
    # Fusion helpers
    # =========================================================================

    @staticmethod
    def _extract_face_contour(face: MediapipeFaceObservation | None) -> tuple[NDArray, NDArray]:
        n_contour = MEDIAPIPE_FACE_CONTOUR_DEFINITION.num_tracked_points
        if face is None or not face.has_detection:
            return np.full((n_contour, 3), np.nan), np.zeros(n_contour)
        return face.face_contour_landmarks_xyz.copy(), face.face_contour_visibility.copy()

    @staticmethod
    def _fuse_body_with_face(body_xyz: NDArray, face: MediapipeFaceObservation | None) -> None:
        if face is None or not face.has_detection:
            return
        face_xyz = face.face_landmarks_xyz
        for pose_idx, face_idx in FACE_TO_POSE_DIRECT_MAP.items():
            if face_idx < face_xyz.shape[0] and not np.isnan(face_xyz[face_idx]).any():
                body_xyz[pose_idx] = face_xyz[face_idx]
        for pose_idx, iris_indices in IRIS_TO_POSE_MAP.items():
            iris_points = face_xyz[iris_indices]
            if not np.isnan(iris_points).any():
                body_xyz[pose_idx] = np.mean(iris_points, axis=0)

    @staticmethod
    def _fuse_body_with_hand(body_xyz: NDArray, hand_xyz: NDArray, hand_to_pose_map: dict[int, int], wrist_pair: tuple[int, int]) -> None:
        if np.isnan(hand_xyz).all():
            return
        pose_wrist_idx, hand_wrist_idx = wrist_pair
        if not np.isnan(hand_xyz[hand_wrist_idx]).any() and not np.isnan(body_xyz[pose_wrist_idx]).any():
            body_xyz[pose_wrist_idx] = (body_xyz[pose_wrist_idx] + hand_xyz[hand_wrist_idx]) / 2.0
        for pose_idx, hand_idx in hand_to_pose_map.items():
            if not np.isnan(hand_xyz[hand_idx]).any():
                body_xyz[pose_idx] = hand_xyz[hand_idx]

    # =========================================================================
    # PointCloud slice accessors — zero-copy views
    # =========================================================================

    @property
    def body_landmarks_xyz(self) -> NDArray:
        return self.points.xyz[_BODY_START:_BODY_END]

    @property
    def body_world_landmarks_xyz(self) -> NDArray:
        if self.pose is None or not self.pose.has_detection:
            return np.full((NUM_POSE_LANDMARKS, 3), np.nan)
        return self.pose.body_world_landmarks_xyz

    @property
    def body_visibility(self) -> NDArray:
        return self.points.visibility[_BODY_START:_BODY_END]

    @property
    def right_hand_landmarks_xyz(self) -> NDArray:
        return self.points.xyz[_RHAND_START:_RHAND_END]

    @property
    def left_hand_landmarks_xyz(self) -> NDArray:
        return self.points.xyz[_LHAND_START:_LHAND_END]

    @property
    def face_landmarks_xyz(self) -> NDArray:
        if self.face is None or not self.face.has_detection:
            return np.full((NUM_FACE_LANDMARKS, 3), np.nan)
        return self.face.face_landmarks_xyz

    @property
    def face_contour_landmarks_xyz(self) -> NDArray:
        return self.points.xyz[_FACE_START:_FACE_END]

    @property
    def face_blendshapes(self) -> dict[str, float] | None:
        if self.face is None:
            return None
        return self.face.face_blendshapes

    @property
    def segmentation_mask(self) -> np.ndarray | None:
        if self.pose is None:
            return None
        return self.pose.segmentation_mask

    @property
    def fused_body_landmarks_xyz(self) -> NDArray:
        return self.body_landmarks_xyz

    # =========================================================================
    # Name accessors
    # =========================================================================

    @property
    def body_landmark_names(self) -> list[str]:
        return list(MEDIAPIPE_BODY_DEFINITION.tracked_points)

    @property
    def right_hand_landmark_names(self) -> list[str]:
        return list(_HOLISTIC_NAMES[_RHAND_START:_RHAND_END])

    @property
    def left_hand_landmark_names(self) -> list[str]:
        return list(_HOLISTIC_NAMES[_LHAND_START:_LHAND_END])

    @property
    def face_contour_landmark_names(self) -> list[str]:
        return list(MEDIAPIPE_FACE_CONTOUR_DEFINITION.tracked_points)

    @property
    def num_face_tesselation_points(self) -> int:
        return NUM_FACE_LANDMARKS

    # =========================================================================
    # Detection state
    # =========================================================================

    @property
    def has_pose(self) -> bool:
        return self.pose is not None and self.pose.has_detection

    @property
    def has_right_hand(self) -> bool:
        return self.hands is not None and self.hands.has_right_hand

    @property
    def has_left_hand(self) -> bool:
        return self.hands is not None and self.hands.has_left_hand

    @property
    def has_face(self) -> bool:
        return self.face is not None and self.face.has_detection

    # =========================================================================
    # Legacy compatibility — all_points()
    # =========================================================================

    def all_points(self, dimensions: int, face_type: str = "contour", scale_by: float = 1.0) -> dict[str, tuple]:
        """
        Get all valid tracked points as {name: (x, y[, z])} dict.
        """
        if dimensions not in (2, 3):
            raise ValueError(f"Invalid dimensions: {dimensions}")

        if face_type == "contour":
            return self.points.to_scaled_tuples(dimensions=dimensions, scale_by=scale_by)

        elif face_type == "tesselation":
            body_rh_lh = self.points.xyz[:_FACE_START] * scale_by
            body_rh_lh_names = self.points.names[:_FACE_START]

            face_full_xyz = self.face_landmarks_xyz * scale_by
            face_full_names = MEDIAPIPE_FACE_TESSELATED_DEFINITION.tracked_points

            result: dict[str, tuple] = {}
            for i, name in enumerate(body_rh_lh_names):
                pt = body_rh_lh[i, :dimensions]
                if not np.isnan(pt).any():
                    result[name] = tuple(pt)
            for i, name in enumerate(face_full_names):
                if i < face_full_xyz.shape[0]:
                    pt = face_full_xyz[i, :dimensions]
                    if not np.isnan(pt).any():
                        result[name] = tuple(pt)
            return result

        else:
            raise ValueError(f"Invalid face type: {face_type}")
