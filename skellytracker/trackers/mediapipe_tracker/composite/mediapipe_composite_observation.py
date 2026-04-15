from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation
from skellytracker.trackers.base_tracker.point_cloud import PointCloud
from skellytracker.trackers.mediapipe_tracker.body.mediapipe_pose_observation import MediapipePoseObservation
from skellytracker.trackers.mediapipe_tracker.face.get_mediapipe_face_info import (
    MEDIAPIPE_FACE_CONTOURS_NAMES,
)
from skellytracker.trackers.mediapipe_tracker.face.mediapipe_face_observation import MediapipeFaceObservation
from skellytracker.trackers.mediapipe_tracker.hands.mediapipe_hand_observation import MediapipeHandObservation
from skellytracker.trackers.mediapipe_tracker.mediapipe_names import (
    FACE_TO_POSE_DIRECT_MAP,
    IRIS_TO_POSE_MAP,
    LEFT_HAND_LANDMARK_NAMES,
    LEFT_HAND_TO_POSE_MAP,
    NUM_FACE_LANDMARKS,
    NUM_HAND_LANDMARKS,
    NUM_POSE_LANDMARKS,
    POSE_LANDMARK_NAMES,
    POSE_LEFT_WRIST_FUSE_WITH_HAND_WRIST,
    POSE_RIGHT_WRIST_FUSE_WITH_HAND_WRIST,
    RIGHT_HAND_LANDMARK_NAMES,
    RIGHT_HAND_TO_POSE_MAP,
)

# Slice boundaries for the concatenated PointCloud:
#   [0:33]        body (fused)
#   [33:54]       right hand
#   [54:75]       left hand
#   [75:75+N]     face contour
_BODY_START = 0
_BODY_END = NUM_POSE_LANDMARKS
_RHAND_START = _BODY_END
_RHAND_END = _RHAND_START + NUM_HAND_LANDMARKS
_LHAND_START = _RHAND_END
_LHAND_END = _LHAND_START + NUM_HAND_LANDMARKS
_FACE_START = _LHAND_END

# Canonical name order — built once at module load
_ALL_NAMES: tuple[str, ...] = (
    tuple(POSE_LANDMARK_NAMES)
    + tuple(RIGHT_HAND_LANDMARK_NAMES)
    + tuple(LEFT_HAND_LANDMARK_NAMES)
    + tuple(MEDIAPIPE_FACE_CONTOURS_NAMES)
)
_FACE_END = len(_ALL_NAMES)


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

    All landmark data is stored in a single PointCloud where names and
    coordinates are structurally coupled — impossible to desync.

    Body landmarks are FUSED at construction time: higher-precision
    hand/face data is spliced in where available.
    """

    tracker_type: str = field(default="mediapipe_composite", init=False)
    frame_number: int = 0
    image_size: tuple[int, int] = (0, 0)  # (height, width)

    # The single source of truth for all landmark data
    points: PointCloud = field(default_factory=lambda: PointCloud.empty(_ALL_NAMES))

    # Sub-observations retained for metadata (blendshapes, segmentation, world coords)
    pose: MediapipePoseObservation | None = None
    hands: MediapipeHandObservation | None = None
    face: MediapipeFaceObservation | None = None

    # ROI boxes used for hand/face crops
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

        # Fuse body landmarks with hand/face data (in-place on body_xyz)
        cls._fuse_body_with_face(body_xyz=body_xyz, face=face)
        cls._fuse_body_with_hand(body_xyz=body_xyz, hand_xyz=rh_xyz, hand_to_pose_map=RIGHT_HAND_TO_POSE_MAP, wrist_pair=POSE_RIGHT_WRIST_FUSE_WITH_HAND_WRIST)
        cls._fuse_body_with_hand(body_xyz=body_xyz, hand_xyz=lh_xyz, hand_to_pose_map=LEFT_HAND_TO_POSE_MAP, wrist_pair=POSE_LEFT_WRIST_FUSE_WITH_HAND_WRIST)

        xyz = np.concatenate([body_xyz, rh_xyz, lh_xyz, face_contour_xyz], axis=0)
        vis = np.concatenate([body_vis, rh_vis, lh_vis, face_contour_vis], axis=0)
        cloud = PointCloud(names=_ALL_NAMES, xyz=xyz, visibility=vis)

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
    # Fusion helpers (static — operate on arrays in-place)
    # =========================================================================

    @staticmethod
    def _extract_face_contour(face: MediapipeFaceObservation | None) -> tuple[NDArray, NDArray]:
        n_contour = len(MEDIAPIPE_FACE_CONTOURS_NAMES)
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

    # Fused body landmarks ARE the body slice — fusion happened at construction
    @property
    def fused_body_landmarks_xyz(self) -> NDArray:
        return self.body_landmarks_xyz

    # =========================================================================
    # Name accessors
    # =========================================================================

    @property
    def body_landmark_names(self) -> list[str]:
        return POSE_LANDMARK_NAMES

    @property
    def right_hand_landmark_names(self) -> list[str]:
        return RIGHT_HAND_LANDMARK_NAMES

    @property
    def left_hand_landmark_names(self) -> list[str]:
        return LEFT_HAND_LANDMARK_NAMES

    @property
    def face_contour_landmark_names(self) -> list[str]:
        return list(MEDIAPIPE_FACE_CONTOURS_NAMES)

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

        Matches the legacy MediapipeObservation.all_points() interface.
        """
        if dimensions not in (2, 3):
            raise ValueError(f"Invalid dimensions: {dimensions}")

        if face_type == "contour":
            return self.points.to_scaled_tuples(dimensions=dimensions, scale_by=scale_by)

        elif face_type == "tesselation":
            body_rh_lh = self.points.xyz[:_FACE_START] * scale_by
            body_rh_lh_names = self.points.names[:_FACE_START]

            face_full_xyz = self.face_landmarks_xyz * scale_by
            face_full_names = tuple(f"face_{i:04d}" for i in range(NUM_FACE_LANDMARKS))

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
