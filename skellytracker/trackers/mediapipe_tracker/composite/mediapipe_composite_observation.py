import numpy as np
from numpydantic import NDArray, Shape
from pydantic import ConfigDict

from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseObservation,
    TrackedPoint2dArray,
    TrackedPointIdString,
    TrackerTypeString,
)
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
from skellytracker.trackers.mediapipe_tracker.body.mediapipe_pose_observation import MediapipePoseObservation


class ROIBox:
    """Bounding box for an ROI crop in full-image coordinates."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Returns (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)


class MediapipeCompositeObservation(BaseObservation):
    """
    Holistic-style observation combining pose, hand, and face detections.

    Stores the individual sub-observations and provides unified access to
    all landmarks, plus fused body landmarks that splice in higher-precision
    hand/face data where available.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    tracker_type: TrackerTypeString = "mediapipe_composite"
    frame_number: int
    image_size: tuple[int, int]  # (height, width)

    pose: MediapipePoseObservation | None
    hands: MediapipeHandObservation | None
    face: MediapipeFaceObservation | None

    # ROI boxes used for hand/face crops (None if full-image detection was used or skipped)
    left_hand_roi: ROIBox | None = None
    right_hand_roi: ROIBox | None = None
    face_roi: ROIBox | None = None

    @classmethod
    def from_detection_results(cls, **kwargs: object) -> "MediapipeCompositeObservation":
        """Not used — MediapipeCompositeObservation is built directly by the composite detector."""
        raise NotImplementedError("Use the MediapipeCompositeDetector to build this observation")

    # =========================================================================
    # Convenience accessors
    # =========================================================================

    @property
    def body_landmarks_xyz(self) -> NDArray[Shape["33, 3"], float]:
        if self.pose is None or not self.pose.has_detection:
            return np.full((NUM_POSE_LANDMARKS, 3), np.nan)
        return self.pose.body_landmarks_xyz

    @property
    def body_world_landmarks_xyz(self) -> NDArray[Shape["33, 3"], float]:
        if self.pose is None or not self.pose.has_detection:
            return np.full((NUM_POSE_LANDMARKS, 3), np.nan)
        return self.pose.body_world_landmarks_xyz

    @property
    def body_visibility(self) -> NDArray[Shape["33"], float]:
        if self.pose is None:
            return np.zeros(NUM_POSE_LANDMARKS)
        return self.pose.body_visibility

    @property
    def right_hand_landmarks_xyz(self) -> NDArray[Shape["21, 3"], float]:
        if self.hands is None or not self.hands.has_right_hand:
            return np.full((NUM_HAND_LANDMARKS, 3), np.nan)
        return self.hands.right_hand_landmarks_xyz

    @property
    def left_hand_landmarks_xyz(self) -> NDArray[Shape["21, 3"], float]:
        if self.hands is None or not self.hands.has_left_hand:
            return np.full((NUM_HAND_LANDMARKS, 3), np.nan)
        return self.hands.left_hand_landmarks_xyz

    @property
    def face_landmarks_xyz(self) -> NDArray[Shape["478, 3"], float]:
        if self.face is None or not self.face.has_detection:
            return np.full((NUM_FACE_LANDMARKS, 3), np.nan)
        return self.face.face_landmarks_xyz

    @property
    def face_contour_landmarks_xyz(self) -> NDArray[Shape["*, 3"], float]:
        if self.face is None or not self.face.has_detection:
            return np.full((len(MEDIAPIPE_FACE_CONTOURS_NAMES), 3), np.nan)
        return self.face.face_contour_landmarks_xyz

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

    # =========================================================================
    # FUSED BODY LANDMARKS
    # =========================================================================

    @property
    def fused_body_landmarks_xyz(self) -> NDArray[Shape["33, 3"], float]:
        """
        Body landmarks with higher-precision hand/face data spliced in.

        Replacements when face is detected:
          - Nose (idx 0): face mesh nose tip
          - Left/right eye (idx 2, 5): mean of iris contour points (pupil center)
          - Eye inner (idx 1, 4): tear duct from face mesh
          - Eye outer (idx 3, 6): outer lid corner from face mesh
          - Ears (idx 7, 8): face mesh ear points
          - Mouth corners (idx 9, 10): face mesh mouth corners

        Replacements when hands are detected:
          - Wrists (idx 15, 16): averaged with hand wrist landmark
          - Pinky/Index/Thumb (idx 17-22): replaced by hand MCP/CMC landmarks
        """
        body = self.body_landmarks_xyz.copy()

        if np.isnan(body).all():
            return body

        # Splice in face data
        if self.face is not None and self.face.has_detection:
            face_xyz = self.face.face_landmarks_xyz

            # Direct replacements: nose, eye corners, ears, mouth
            for pose_idx, face_idx in FACE_TO_POSE_DIRECT_MAP.items():
                if face_idx < face_xyz.shape[0] and not np.isnan(face_xyz[face_idx]).any():
                    body[pose_idx] = face_xyz[face_idx]

            # Iris centroid replacements: left_eye and right_eye → mean of iris contour
            for pose_idx, iris_indices in IRIS_TO_POSE_MAP.items():
                iris_points = face_xyz[iris_indices]
                if not np.isnan(iris_points).any():
                    body[pose_idx] = np.mean(iris_points, axis=0)

        # Splice in right hand data
        if self.hands is not None and self.hands.has_right_hand:
            rh = self.hands.right_hand_landmarks_xyz

            # Average wrist position
            pose_wrist_idx, hand_wrist_idx = POSE_RIGHT_WRIST_FUSE_WITH_HAND_WRIST
            if not np.isnan(rh[hand_wrist_idx]).any() and not np.isnan(body[pose_wrist_idx]).any():
                body[pose_wrist_idx] = (body[pose_wrist_idx] + rh[hand_wrist_idx]) / 2.0

            # Direct replacements: pinky, index, thumb
            for pose_idx, hand_idx in RIGHT_HAND_TO_POSE_MAP.items():
                if not np.isnan(rh[hand_idx]).any():
                    body[pose_idx] = rh[hand_idx]

        # Splice in left hand data
        if self.hands is not None and self.hands.has_left_hand:
            lh = self.hands.left_hand_landmarks_xyz

            # Average wrist position
            pose_wrist_idx, hand_wrist_idx = POSE_LEFT_WRIST_FUSE_WITH_HAND_WRIST
            if not np.isnan(lh[hand_wrist_idx]).any() and not np.isnan(body[pose_wrist_idx]).any():
                body[pose_wrist_idx] = (body[pose_wrist_idx] + lh[hand_wrist_idx]) / 2.0

            # Direct replacements: pinky, index, thumb
            for pose_idx, hand_idx in LEFT_HAND_TO_POSE_MAP.items():
                if not np.isnan(lh[hand_idx]).any():
                    body[pose_idx] = lh[hand_idx]

        return body

    # =========================================================================
    # BaseObservation interface
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

    def get_confidence_scores(self) -> NDArray[Shape["*"], float]:
        body_vis = self.body_visibility
        right_hand_vis = self.hands.right_hand_visibility if self.hands is not None else np.zeros(NUM_HAND_LANDMARKS)
        left_hand_vis = self.hands.left_hand_visibility if self.hands is not None else np.zeros(NUM_HAND_LANDMARKS)
        face_vis = self.face.face_contour_visibility if self.face is not None else np.zeros(len(MEDIAPIPE_FACE_CONTOURS_NAMES))
        return np.concatenate([body_vis, right_hand_vis, left_hand_vis, face_vis])

    def to_tracked_points(self, *, confidence_threshold: float | None = None) -> dict[TrackedPointIdString, TrackedPoint2dArray]:
        result: dict[TrackedPointIdString, TrackedPoint2dArray] = {}

        # Body points (using fused landmarks)
        fused_body = self.fused_body_landmarks_xyz
        body_vis = self.body_visibility
        for i, name in enumerate(POSE_LANDMARK_NAMES):
            if np.isnan(fused_body[i]).any():
                continue
            if confidence_threshold is not None and body_vis[i] < confidence_threshold:
                continue
            result[name] = np.array(fused_body[i, :2])

        # Right hand
        rh = self.right_hand_landmarks_xyz
        rh_vis = self.hands.right_hand_visibility if self.hands is not None else np.zeros(NUM_HAND_LANDMARKS)
        for i, name in enumerate(RIGHT_HAND_LANDMARK_NAMES):
            if np.isnan(rh[i]).any():
                continue
            if confidence_threshold is not None and rh_vis[i] < confidence_threshold:
                continue
            result[name] = np.array(rh[i, :2])

        # Left hand
        lh = self.left_hand_landmarks_xyz
        lh_vis = self.hands.left_hand_visibility if self.hands is not None else np.zeros(NUM_HAND_LANDMARKS)
        for i, name in enumerate(LEFT_HAND_LANDMARK_NAMES):
            if np.isnan(lh[i]).any():
                continue
            if confidence_threshold is not None and lh_vis[i] < confidence_threshold:
                continue
            result[name] = np.array(lh[i, :2])

        # Face contours
        fc = self.face_contour_landmarks_xyz
        fc_vis = self.face.face_contour_visibility if self.face is not None else np.zeros(len(MEDIAPIPE_FACE_CONTOURS_NAMES))
        for i, name in enumerate(MEDIAPIPE_FACE_CONTOURS_NAMES):
            if np.isnan(fc[i]).any():
                continue
            if confidence_threshold is not None and fc_vis[i] < confidence_threshold:
                continue
            result[name] = np.array(fc[i, :2])

        return result

    def to_2d_array(self, *, confidence_threshold: float | None = None, fill_with_nans: bool = True) -> NDArray[Shape["*, 2"], float]:
        """Concatenate body + right hand + left hand + face contour as 2D array."""
        points_2d = np.concatenate(
            [
                self.fused_body_landmarks_xyz[:, :2],
                self.right_hand_landmarks_xyz[:, :2],
                self.left_hand_landmarks_xyz[:, :2],
                self.face_contour_landmarks_xyz[:, :2],
            ],
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
