import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, LandmarkList
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions.face_mesh import FACEMESH_NUM_LANDMARKS_WITH_IRISES

from skellytracker.old.base_tracker.base_tracker_abcs import BaseObservation, TrackedPoint2dArray
from skellytracker.old.base_tracker.point_cloud import PointCloud
from skellytracker.old.legacy_mediapipe_tracker.get_legacy_mediapipe_face_info import \
    MEDIAPIPE_FACE_CONTOURS_INDICIES, \
    MEDIAPIPE_FACE_CONTOURS_NAMES

logger = logging.getLogger(__name__)


@runtime_checkable
class LegacyMediapipeResults(Protocol):
    """Structural type describing the object returned by mp.solutions.holistic.Holistic.process().
    MediaPipe does not export this type publicly, so we define a Protocol matching the fields we use."""
    pose_landmarks: NormalizedLandmarkList | None
    pose_world_landmarks: LandmarkList | None
    right_hand_landmarks: NormalizedLandmarkList | None
    left_hand_landmarks: NormalizedLandmarkList | None
    face_landmarks: NormalizedLandmarkList | None
    segmentation_mask: np.ndarray | None


@dataclass(slots=True)
class LegacyMediapipeObservation(BaseObservation):
    tracker_type: str = field(default="legacy_mediapipe", init=False)
    frame_number: int = 0
    pose_landmarks: NormalizedLandmarkList | None = None
    pose_world_landmarks: LandmarkList | None = None
    right_hand_landmarks: NormalizedLandmarkList | None = None
    left_hand_landmarks: NormalizedLandmarkList | None = None
    face_landmarks: NormalizedLandmarkList | None = None
    segmentation_mask: np.ndarray | None = None
    image_size: tuple[int, int] = (0, 0)

    # Required by BaseObservation; not populated for the legacy tracker —
    # all concrete methods are overridden below and do not delegate to PointCloud.
    points: PointCloud = field(default_factory=lambda: PointCloud.empty(()))

    @classmethod
    def from_detection_results(
        cls,
        frame_number: int,
        mediapipe_results: LegacyMediapipeResults,
        image_size: tuple[int, int],
        include_segmentation_mask: bool = True,
    ) -> "LegacyMediapipeObservation":
        if include_segmentation_mask:
            segmentation_mask = mediapipe_results.segmentation_mask
        else:
            segmentation_mask = None
        return cls(
            frame_number=frame_number,
            pose_landmarks=mediapipe_results.pose_landmarks,
            pose_world_landmarks=mediapipe_results.pose_world_landmarks,
            right_hand_landmarks=mediapipe_results.right_hand_landmarks,
            left_hand_landmarks=mediapipe_results.left_hand_landmarks,
            face_landmarks=mediapipe_results.face_landmarks,
            segmentation_mask=segmentation_mask,
            image_size=image_size,
        )

    @property
    def body_landmark_names(self) -> list[str]:
        return [f"body.{landmark.name.lower()}" for landmark in mp_holistic.PoseLandmark]

    @property
    def hand_landmark_names(self) -> list[str]:
        return [landmark.name.lower() for landmark in mp_holistic.HandLandmark]

    @property
    def right_hand_landmark_names(self) -> list[str]:
        return [f"right_hand.{landmark}" for landmark in self.hand_landmark_names]

    @property
    def left_hand_landmark_names(self) -> list[str]:
        return [f"left_hand.{landmark}" for landmark in self.hand_landmark_names]

    @property
    def face_contour_landmark_names(self) -> list[str]:
        return MEDIAPIPE_FACE_CONTOURS_NAMES

    @property
    def num_body_points(self) -> int:
        return len(self.body_landmark_names)

    @property
    def num_single_hand_points(self) -> int:
        return len(self.hand_landmark_names)

    @property
    def num_face_tesselation_points(self) -> int:
        return FACEMESH_NUM_LANDMARKS_WITH_IRISES

    @property
    def num_face_contour_points(self) -> int:
        return len(MEDIAPIPE_FACE_CONTOURS_INDICIES)

    @property
    def num_total_points(self) -> int:
        return self.num_body_points + (2 * self.num_single_hand_points) + self.num_face_contour_points

    @property
    def body_points_xyz(self) -> np.ndarray:
        if self.pose_landmarks is None:
            return np.full((self.num_body_points, 3), np.nan)
        return self._landmarks_to_array(self.pose_landmarks)

    @property
    def right_hand_points_xyz(self) -> np.ndarray:
        if self.right_hand_landmarks is None:
            return np.full((self.num_single_hand_points, 3), np.nan)
        return self._landmarks_to_array(self.right_hand_landmarks)

    @property
    def left_hand_points_xyz(self) -> np.ndarray:
        if self.left_hand_landmarks is None:
            return np.full((self.num_single_hand_points, 3), np.nan)
        return self._landmarks_to_array(self.left_hand_landmarks)

    @property
    def face_tesselation_points_xyz(self) -> np.ndarray:
        if self.face_landmarks is None:
            return np.full((self.num_face_tesselation_points, 3), np.nan)

        landmarks = self._landmarks_to_array(self.face_landmarks)

        # Pad with NaN if iris landmarks are missing
        expected_count = self.num_face_tesselation_points
        actual_count = landmarks.shape[0]
        if actual_count < expected_count:
            padding = np.full((expected_count - actual_count, 3), np.nan)
            landmarks = np.vstack([landmarks, padding])

        return landmarks

    @property
    def face_contour_points_xyz(self) -> np.ndarray:
        all_face_landmarks = self.face_tesselation_points_xyz
        if np.isnan(all_face_landmarks).all():
            return np.full((self.num_face_contour_points, 3), np.nan)
        indices = [int(name.rsplit("_", 1)[1]) for name in self.face_contour_landmark_names]
        return all_face_landmarks[indices]

    @property
    def has_pose(self) -> bool:
        return self.pose_landmarks is not None

    @property
    def has_right_hand(self) -> bool:
        return self.right_hand_landmarks is not None

    @property
    def has_left_hand(self) -> bool:
        return self.left_hand_landmarks is not None

    @property
    def has_face(self) -> bool:
        return self.face_landmarks is not None

    @property
    def body_visibility(self) -> np.ndarray:
        """Extract visibility scores from body landmarks."""
        if self.pose_landmarks is None:
            return np.full(self.num_body_points, 0.0)
        return np.array([
            landmark.visibility if hasattr(landmark, "visibility") else 1.0
            for landmark in self.pose_landmarks.landmark
        ])

    @property
    def hand_visibility(self) -> np.ndarray:
        """Hands have no per-landmark visibility in the legacy API; return all 1.0."""
        return np.full(self.num_single_hand_points, 1.0)

    @property
    def face_visibility(self) -> np.ndarray:
        """Extract visibility (presence) scores for face contour points."""
        if self.face_landmarks is None:
            return np.full(self.num_face_contour_points, 0.0)
        if np.isnan(self.face_tesselation_points_xyz).all():
            return np.full(self.num_face_contour_points, 0.0)

        face_contour_indices = [int(name.rsplit("_", 1)[1]) for name in self.face_contour_landmark_names]
        visibilities = []
        for idx in face_contour_indices:
            if idx < len(self.face_landmarks.landmark):
                landmark = self.face_landmarks.landmark[idx]
                visibility = landmark.presence if hasattr(landmark, "presence") else 1.0
            else:
                visibility = 0.0
            visibilities.append(visibility)
        return np.array(visibilities)

    def _landmarks_to_array(self, landmarks: NormalizedLandmarkList) -> np.ndarray:
        landmark_array = np.array([
            (landmark.x, landmark.y, landmark.z)
            for landmark in landmarks.landmark
        ])
        # Convert from normalized image coordinates to pixel coordinates.
        # Z is multiplied by image width per MediaPipe docs.
        landmark_array *= np.array([self.image_size[1], self.image_size[0], self.image_size[1]])
        return landmark_array

    def get_confidence_scores(self) -> np.ndarray:
        """Get visibility/confidence scores for all tracked points."""
        return np.concatenate([
            self.body_visibility,
            self.hand_visibility,   # right hand
            self.hand_visibility,   # left hand (same constant array)
            self.face_visibility,
        ])

    def to_2d_array(self, *, confidence_threshold: float | None = None, fill_with_nans: bool = True) -> np.ndarray:
        """Convert to (N, 2) array with optional confidence filtering."""
        points_2d = np.concatenate(
            (
                self.body_points_xyz[..., :2],
                self.right_hand_points_xyz[..., :2],
                self.left_hand_points_xyz[..., :2],
                self.face_contour_points_xyz[..., :2],
            ),
            axis=0,
        )
        if confidence_threshold is not None:
            confidence_scores = self.get_confidence_scores()
            points_2d = self.filter_by_confidence(
                points=points_2d,
                confidence_scores=confidence_scores,
                confidence_threshold=confidence_threshold,
                fill_with_nans=fill_with_nans,
            )
        return points_2d

    def to_3d_array(self, *, confidence_threshold: float | None = None, fill_with_nans: bool = True) -> np.ndarray:
        """Convert to (N, 3) array with optional confidence filtering."""
        points_3d = np.concatenate(
            (
                self.body_points_xyz,
                self.right_hand_points_xyz,
                self.left_hand_points_xyz,
                self.face_contour_points_xyz,
            ),
            axis=0,
        )
        if confidence_threshold is not None:
            confidence_scores = self.get_confidence_scores()
            points_3d = self.filter_by_confidence(
                points=points_3d,
                confidence_scores=confidence_scores,
                confidence_threshold=confidence_threshold,
                fill_with_nans=fill_with_nans,
            )
        return points_3d

    def to_tracked_points(self, *, confidence_threshold: float | None = None) -> dict[str, TrackedPoint2dArray]:
        """Get tracked points filtered by confidence."""
        points = self.all_points(dimensions=2)

        if confidence_threshold is not None:
            confidence_scores = self.get_confidence_scores()
            all_names = (
                self.body_landmark_names
                + self.right_hand_landmark_names
                + self.left_hand_landmark_names
                + self.face_contour_landmark_names
            )
            filtered_points: dict[str, TrackedPoint2dArray] = {}
            for i, name in enumerate(all_names):
                if confidence_scores[i] >= confidence_threshold and name in points:
                    filtered_points[name] = np.array(points[name])
            return filtered_points

        return {name: np.array([x, y]) for name, (x, y) in points.items()}

    def all_points(self, dimensions: int, face_type: str = "contour", scale_by: float = 1.0) -> dict[str, tuple]:
        if dimensions not in (2, 3):
            raise ValueError(f"Invalid dimensions: {dimensions}")

        body_xyz = self.body_points_xyz.copy() * scale_by
        right_hand_xyz = self.right_hand_points_xyz.copy() * scale_by
        left_hand_xyz = self.left_hand_points_xyz.copy() * scale_by

        if face_type == "tesselation":
            face_xyz = self.face_tesselation_points_xyz.copy() * scale_by
        elif face_type == "contour":
            face_xyz = self.face_contour_points_xyz.copy() * scale_by
        else:
            raise ValueError(f"Invalid face type: {face_type}")

        all_points_by_name: dict[str, tuple] = {}
        for index, point_name in enumerate(self.body_landmark_names):
            all_points_by_name[point_name] = tuple(body_xyz[index, :dimensions])
        for index, point_name in enumerate(self.right_hand_landmark_names):
            all_points_by_name[point_name] = tuple(right_hand_xyz[index, :dimensions])
        for index, point_name in enumerate(self.left_hand_landmark_names):
            all_points_by_name[point_name] = tuple(left_hand_xyz[index, :dimensions])
        for index, point_name in enumerate(self.face_contour_landmark_names):
            all_points_by_name[point_name] = tuple(face_xyz[index, :dimensions])
        return all_points_by_name


LegacyMediapipeObservations = list[LegacyMediapipeObservation]
