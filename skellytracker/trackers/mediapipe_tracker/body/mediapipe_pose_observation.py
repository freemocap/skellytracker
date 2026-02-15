import numpy as np
from numpydantic import NDArray, Shape
from pydantic import ConfigDict

from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseObservation,
    TrackedPoint2dArray,
    TrackedPointIdString,
    TrackerTypeString,
)
from skellytracker.trackers.mediapipe_tracker.mediapipe_names import (
    NUM_POSE_LANDMARKS,
    POSE_LANDMARK_NAMES,
)


class MediapipePoseObservation(BaseObservation):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tracker_type: TrackerTypeString = "mediapipe_pose"
    frame_number: int
    image_size: tuple[int, int]  # (height, width)

    body_landmarks_xyz: NDArray[Shape["33, 3"], float]
    body_world_landmarks_xyz: NDArray[Shape["33, 3"], float]
    body_visibility: NDArray[Shape["33"], float]
    segmentation_mask: np.ndarray | None

    @classmethod
    def from_detection_results(
        cls,
        frame_number: int,
        pose_landmarker_result: "mp.tasks.vision.PoseLandmarkerResult",
        image_size: tuple[int, int],
    ) -> "MediapipePoseObservation":
        """
        Convert a PoseLandmarkerResult into a MediapipePoseObservation.

        Expects exactly one pose detected (num_poses=1). If no pose detected,
        returns all-NaN arrays.
        """
        height, width = image_size

        if len(pose_landmarker_result.pose_landmarks) == 0:
            return cls(
                frame_number=frame_number,
                image_size=image_size,
                body_landmarks_xyz=np.full((NUM_POSE_LANDMARKS, 3), np.nan),
                body_world_landmarks_xyz=np.full((NUM_POSE_LANDMARKS, 3), np.nan),
                body_visibility=np.zeros(NUM_POSE_LANDMARKS),
                segmentation_mask=None,
            )

        # Take the first (and typically only) detected pose
        landmarks = pose_landmarker_result.pose_landmarks[0]
        world_landmarks = pose_landmarker_result.pose_world_landmarks[0]

        # Normalized landmarks → pixel coordinates
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

        return cls(
            frame_number=frame_number,
            image_size=image_size,
            body_landmarks_xyz=body_xyz,
            body_world_landmarks_xyz=body_world_xyz,
            body_visibility=visibility,
            segmentation_mask=seg_mask,
        )

    @property
    def has_detection(self) -> bool:
        """True if a body was detected (not all NaN)."""
        return not np.isnan(self.body_landmarks_xyz).all()

    def get_confidence_scores(self) -> NDArray[Shape["33"], float]:
        return self.body_visibility

    def to_tracked_points(self, *, confidence_threshold: float | None = None) -> dict[TrackedPointIdString, TrackedPoint2dArray]:
        points_2d = self.body_landmarks_xyz[:, :2]
        result: dict[TrackedPointIdString, TrackedPoint2dArray] = {}
        for i, name in enumerate(POSE_LANDMARK_NAMES):
            if confidence_threshold is not None and self.body_visibility[i] < confidence_threshold:
                continue
            result[name] = np.array(points_2d[i])
        return result

    def to_2d_array(self, *, confidence_threshold: float | None = None, fill_with_nans: bool = True) -> NDArray[Shape["33, 2"], float]:
        points_2d = self.body_landmarks_xyz[:, :2].copy()
        if confidence_threshold is not None:
            points_2d = self.filter_by_confidence(
                points=points_2d,
                confidence_scores=self.body_visibility,
                confidence_threshold=confidence_threshold,
                fill_with_nans=fill_with_nans,
            )
        return points_2d