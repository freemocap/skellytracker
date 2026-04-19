from dataclasses import dataclass, field

import numpy as np
from mediapipe.tasks.python.vision import HandLandmarkerResult
from numpy.typing import NDArray

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation
from skellytracker.trackers.base_tracker.point_cloud import PointCloud
from skellytracker.trackers.mediapipe_tracker.mediapipe_names import (
    LEFT_HAND_LANDMARK_NAMES,
    NUM_HAND_LANDMARKS,
    RIGHT_HAND_LANDMARK_NAMES,
)

_RIGHT_HAND_NAMES: tuple[str, ...] = tuple(RIGHT_HAND_LANDMARK_NAMES)
_LEFT_HAND_NAMES: tuple[str, ...] = tuple(LEFT_HAND_LANDMARK_NAMES)
_ALL_HAND_NAMES: tuple[str, ...] = _RIGHT_HAND_NAMES + _LEFT_HAND_NAMES


@dataclass(slots=True)
class MediapipeHandObservation(BaseObservation):
    """
    Hand observation storing left + right hands in a single PointCloud.

    Layout: [0:21] = right hand, [21:42] = left hand.
    """

    tracker_type: str = field(default="mediapipe_hand", init=False)
    frame_number: int = 0
    image_size: tuple[int, int] = (0, 0)

    points: PointCloud = field(default_factory=lambda: PointCloud.empty(_ALL_HAND_NAMES))

    @classmethod
    def from_arrays(
        cls,
        frame_number: int,
        image_size: tuple[int, int],
        right_hand_xyz: NDArray,
        left_hand_xyz: NDArray,
        right_hand_visibility: NDArray,
        left_hand_visibility: NDArray,
    ) -> "MediapipeHandObservation":
        """Build from pre-computed arrays (used by composite detector)."""
        xyz = np.concatenate([right_hand_xyz, left_hand_xyz], axis=0)
        vis = np.concatenate([right_hand_visibility, left_hand_visibility], axis=0)
        cloud = PointCloud(names=_ALL_HAND_NAMES, xyz=xyz, visibility=vis)
        return cls(frame_number=frame_number, image_size=image_size, points=cloud)

    @classmethod
    def from_detection_results(
        cls,
        frame_number: int,
        hand_landmarker_result: HandLandmarkerResult,
        image_size: tuple[int, int],
    ) -> "MediapipeHandObservation":
        """Convert a HandLandmarkerResult into a MediapipeHandObservation."""
        height, width = image_size

        right_xyz = np.full((NUM_HAND_LANDMARKS, 3), np.nan)
        left_xyz = np.full((NUM_HAND_LANDMARKS, 3), np.nan)
        right_vis = np.zeros(NUM_HAND_LANDMARKS)
        left_vis = np.zeros(NUM_HAND_LANDMARKS)

        for i, hand_landmarks in enumerate(hand_landmarker_result.hand_landmarks):
            handedness = hand_landmarker_result.handedness[i]
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

        return cls.from_arrays(
            frame_number=frame_number,
            image_size=image_size,
            right_hand_xyz=right_xyz,
            left_hand_xyz=left_xyz,
            right_hand_visibility=right_vis,
            left_hand_visibility=left_vis,
        )

    @classmethod
    def from_crop_results(
        cls,
        frame_number: int,
        hand_landmarker_result: HandLandmarkerResult,
        crop_origin: tuple[int, int],
        crop_size: tuple[int, int],
        full_image_size: tuple[int, int],
        handedness_hint: str,
    ) -> "MediapipeHandObservation":
        """Convert a HandLandmarkerResult from a crop back to full-image coordinates."""
        crop_h, crop_w = crop_size
        y_off, x_off = crop_origin

        target_xyz = np.full((NUM_HAND_LANDMARKS, 3), np.nan)
        target_vis = np.zeros(NUM_HAND_LANDMARKS)

        if len(hand_landmarker_result.hand_landmarks) > 0:
            best_idx = 0
            for i, handedness in enumerate(hand_landmarker_result.handedness):
                if handedness[0].category_name == handedness_hint:
                    best_idx = i
                    break

            hand_landmarks = hand_landmarker_result.hand_landmarks[best_idx]
            target_xyz = np.array(
                [(lm.x * crop_w + x_off, lm.y * crop_h + y_off, lm.z * crop_w) for lm in hand_landmarks]
            )
            target_vis = np.array(
                [lm.presence if lm.presence is not None else 1.0 for lm in hand_landmarks]
            )

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

        return cls.from_arrays(
            frame_number=frame_number,
            image_size=full_image_size,
            right_hand_xyz=right_xyz,
            left_hand_xyz=left_xyz,
            right_hand_visibility=right_vis,
            left_hand_visibility=left_vis,
        )

    # =========================================================================
    # Convenience accessors — views into the PointCloud
    # =========================================================================

    @property
    def right_hand_landmarks_xyz(self) -> NDArray:
        return self.points.xyz[:NUM_HAND_LANDMARKS]

    @property
    def left_hand_landmarks_xyz(self) -> NDArray:
        return self.points.xyz[NUM_HAND_LANDMARKS:]

    @property
    def right_hand_visibility(self) -> NDArray:
        return self.points.visibility[:NUM_HAND_LANDMARKS]

    @property
    def left_hand_visibility(self) -> NDArray:
        return self.points.visibility[NUM_HAND_LANDMARKS:]

    @property
    def has_right_hand(self) -> bool:
        return not np.isnan(self.right_hand_landmarks_xyz).all()

    @property
    def has_left_hand(self) -> bool:
        return not np.isnan(self.left_hand_landmarks_xyz).all()

    @property
    def has_detection(self) -> bool:
        return self.has_right_hand or self.has_left_hand
