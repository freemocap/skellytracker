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
    LEFT_HAND_LANDMARK_NAMES,
    NUM_HAND_LANDMARKS,
    RIGHT_HAND_LANDMARK_NAMES,
)


class MediapipeHandObservation(BaseObservation):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tracker_type: TrackerTypeString = "mediapipe_hand"
    frame_number: int
    image_size: tuple[int, int]  # (height, width)

    right_hand_landmarks_xyz: NDArray[Shape["21, 3"], float]
    left_hand_landmarks_xyz: NDArray[Shape["21, 3"], float]
    right_hand_visibility: NDArray[Shape["21"], float]
    left_hand_visibility: NDArray[Shape["21"], float]

    @classmethod
    def from_detection_results(
        cls,
        frame_number: int,
        hand_landmarker_result: "mp.tasks.vision.HandLandmarkerResult",
        image_size: tuple[int, int],
    ) -> "MediapipeHandObservation":
        """
        Convert a HandLandmarkerResult into a MediapipeHandObservation.

        Uses handedness labels to assign hands to left/right. If the same
        hand is detected twice, the higher-confidence detection is used.
        """
        height, width = image_size

        right_xyz = np.full((NUM_HAND_LANDMARKS, 3), np.nan)
        left_xyz = np.full((NUM_HAND_LANDMARKS, 3), np.nan)
        right_vis = np.zeros(NUM_HAND_LANDMARKS)
        left_vis = np.zeros(NUM_HAND_LANDMARKS)

        for i, hand_landmarks in enumerate(hand_landmarker_result.hand_landmarks):
            handedness = hand_landmarker_result.handedness[i]
            # handedness[0].category_name is "Left" or "Right"
            # MediaPipe assumes mirrored input, so "Left" from detector = user's left hand
            label = handedness[0].category_name

            landmarks_xyz = np.array(
                [(lm.x * width, lm.y * height, lm.z * width) for lm in hand_landmarks]
            )
            visibility = np.array(
                [lm.presence if lm.presence is not None else 1.0 for lm in hand_landmarks]
            )

            if label == "Left":
                left_xyz = landmarks_xyz
                left_vis = visibility
            elif label == "Right":
                right_xyz = landmarks_xyz
                right_vis = visibility

        return cls(
            frame_number=frame_number,
            image_size=image_size,
            right_hand_landmarks_xyz=right_xyz,
            left_hand_landmarks_xyz=left_xyz,
            right_hand_visibility=right_vis,
            left_hand_visibility=left_vis,
        )

    @classmethod
    def from_crop_results(
        cls,
        frame_number: int,
        hand_landmarker_result: "mp.tasks.vision.HandLandmarkerResult",
        crop_origin: tuple[int, int],
        crop_size: tuple[int, int],
        full_image_size: tuple[int, int],
        handedness_hint: str,
    ) -> "MediapipeHandObservation":
        """
        Convert a HandLandmarkerResult from a cropped image back to full-image coordinates.

        Args:
            crop_origin: (y_offset, x_offset) of the crop in the full image.
            crop_size: (crop_height, crop_width) of the crop.
            full_image_size: (full_height, full_width).
            handedness_hint: "Left" or "Right" — which hand we expect to find in this crop.
        """
        crop_h, crop_w = crop_size
        y_off, x_off = crop_origin

        target_xyz = np.full((NUM_HAND_LANDMARKS, 3), np.nan)
        target_vis = np.zeros(NUM_HAND_LANDMARKS)

        if len(hand_landmarker_result.hand_landmarks) > 0:
            # Take the best detection from this crop
            # If multiple hands found, prefer the one matching handedness_hint
            best_idx = 0
            for i, handedness in enumerate(hand_landmarker_result.handedness):
                if handedness[0].category_name == handedness_hint:
                    best_idx = i
                    break

            hand_landmarks = hand_landmarker_result.hand_landmarks[best_idx]

            # Convert from normalized crop coords → full image pixel coords
            landmarks_xyz = np.array(
                [
                    (
                        lm.x * crop_w + x_off,
                        lm.y * crop_h + y_off,
                        lm.z * crop_w,  # z is relative to crop width
                    )
                    for lm in hand_landmarks
                ]
            )
            visibility = np.array(
                [lm.presence if lm.presence is not None else 1.0 for lm in hand_landmarks]
            )
            target_xyz = landmarks_xyz
            target_vis = visibility

        # Build observation with only the relevant hand populated
        right_xyz = np.full((NUM_HAND_LANDMARKS, 3), np.nan)
        left_xyz = np.full((NUM_HAND_LANDMARKS, 3), np.nan)
        right_vis = np.zeros(NUM_HAND_LANDMARKS)
        left_vis = np.zeros(NUM_HAND_LANDMARKS)

        if handedness_hint == "Right":
            right_xyz = target_xyz
            right_vis = target_vis
        else:
            left_xyz = target_xyz
            left_vis = target_vis

        return cls(
            frame_number=frame_number,
            image_size=full_image_size,
            right_hand_landmarks_xyz=right_xyz,
            left_hand_landmarks_xyz=left_xyz,
            right_hand_visibility=right_vis,
            left_hand_visibility=left_vis,
        )

    @property
    def has_right_hand(self) -> bool:
        return not np.isnan(self.right_hand_landmarks_xyz).all()

    @property
    def has_left_hand(self) -> bool:
        return not np.isnan(self.left_hand_landmarks_xyz).all()

    @property
    def has_detection(self) -> bool:
        return self.has_right_hand or self.has_left_hand

    def get_confidence_scores(self) -> NDArray[Shape["42"], float]:
        return np.concatenate([self.right_hand_visibility, self.left_hand_visibility])

    def to_tracked_points(self, *, confidence_threshold: float | None = None) -> dict[TrackedPointIdString, TrackedPoint2dArray]:
        result: dict[TrackedPointIdString, TrackedPoint2dArray] = {}
        for i, name in enumerate(RIGHT_HAND_LANDMARK_NAMES):
            if np.isnan(self.right_hand_landmarks_xyz[i]).any():
                continue
            if confidence_threshold is not None and self.right_hand_visibility[i] < confidence_threshold:
                continue
            result[name] = np.array(self.right_hand_landmarks_xyz[i, :2])
        for i, name in enumerate(LEFT_HAND_LANDMARK_NAMES):
            if np.isnan(self.left_hand_landmarks_xyz[i]).any():
                continue
            if confidence_threshold is not None and self.left_hand_visibility[i] < confidence_threshold:
                continue
            result[name] = np.array(self.left_hand_landmarks_xyz[i, :2])
        return result

    def to_2d_array(self, *, confidence_threshold: float | None = None, fill_with_nans: bool = True) -> NDArray[Shape["42, 2"], float]:
        points_2d = np.concatenate(
            [self.right_hand_landmarks_xyz[:, :2], self.left_hand_landmarks_xyz[:, :2]],
            axis=0,
        )
        if confidence_threshold is not None:
            points_2d = self.filter_by_confidence(
                points=points_2d,
                confidence_scores=self.get_confidence_scores(),
                confidence_threshold=confidence_threshold,
                fill_with_nans=fill_with_nans,
            )
        return points_2d
