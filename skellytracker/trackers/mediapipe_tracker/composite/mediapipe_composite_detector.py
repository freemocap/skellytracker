import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetector
from skellytracker.trackers.mediapipe_tracker.body.mediapipe_pose_detector import MediapipePoseDetector
from skellytracker.trackers.mediapipe_tracker.composite.mediapipe_composite_config import \
    MediapipeCompositeDetectorConfig
from skellytracker.trackers.mediapipe_tracker.composite.mediapipe_composite_observation import (
    MediapipeCompositeObservation,
    ROIBox,
)
from skellytracker.trackers.mediapipe_tracker.face.mediapipe_face_detector import MediapipeFaceDetector
from skellytracker.trackers.mediapipe_tracker.face.mediapipe_face_observation import MediapipeFaceObservation
from skellytracker.trackers.mediapipe_tracker.composite.composite_tracker_mappings import (
    HAND_WRIST_INDEX,
    POSE_LEFT_EAR_INDEX,
    POSE_LEFT_ELBOW_INDEX,
    POSE_LEFT_EYE_INDEX,
    POSE_LEFT_EYE_INNER_INDEX,
    POSE_LEFT_EYE_OUTER_INDEX,
    POSE_LEFT_WRIST_INDEX,
    POSE_MOUTH_LEFT_INDEX,
    POSE_MOUTH_RIGHT_INDEX,
    POSE_NOSE_INDEX,
    POSE_RIGHT_EAR_INDEX,
    POSE_RIGHT_ELBOW_INDEX,
    POSE_RIGHT_EYE_INDEX,
    POSE_RIGHT_EYE_INNER_INDEX,
    POSE_RIGHT_EYE_OUTER_INDEX,
    POSE_RIGHT_WRIST_INDEX,
)
from skellytracker.trackers.mediapipe_tracker.hands.mediapipe_hand_detector import MediapipeHandDetector
from skellytracker.trackers.mediapipe_tracker.hands.mediapipe_hand_observation import (
    NUM_HAND_LANDMARKS,
    MediapipeHandObservation,
)
from skellytracker.trackers.mediapipe_tracker.names_and_connections import (
    MEDIAPIPE_HOLISTIC_DEFINITION,
)

logger = logging.getLogger(__name__)

# All pose landmark indices that correspond to head anatomy.
# Used to compute the face ROI bounding box — robust to side-views because
# we take the bbox of whichever subset is actually visible.
HEAD_POSE_INDICES: list[int] = [
    POSE_NOSE_INDEX,
    POSE_LEFT_EYE_INNER_INDEX,
    POSE_LEFT_EYE_INDEX,
    POSE_LEFT_EYE_OUTER_INDEX,
    POSE_RIGHT_EYE_INNER_INDEX,
    POSE_RIGHT_EYE_INDEX,
    POSE_RIGHT_EYE_OUTER_INDEX,
    POSE_LEFT_EAR_INDEX,
    POSE_RIGHT_EAR_INDEX,
    POSE_MOUTH_LEFT_INDEX,
    POSE_MOUTH_RIGHT_INDEX,
]

@dataclass
class MediapipeCompositeDetector(BaseDetector):
    """
    Pose-first, crop-detect pipeline that recreates MediaPipe Holistic behavior.

    Pipeline:
    1. Run PoseLandmarker on the full image to get body landmarks.
    2. If body is detected, compute ROI crops for hands and face based on
       pose landmark positions.
    3. Run HandLandmarker and FaceLandmarker on the cropped regions.
    4. Transform sub-detection coordinates back to full-image space.
    5. Merge all results into a MediapipeCompositeObservation.

    If a pose anchor landmark (e.g. wrist for hands, nose for face) has low
    visibility, falls back to running that sub-detector on the full image.
    If no body is detected at all, returns an all-NaN observation.
    """

    config: MediapipeCompositeDetectorConfig
    pose_detector: MediapipePoseDetector
    hand_detector: MediapipeHandDetector | None
    face_detector: MediapipeFaceDetector | None

    # Last known hand sizes (bounding box diagonal), used for ROI sizing
    # when wrist-to-elbow foreshortens (arm pointing at camera)
    last_left_hand_size: float = 0.0
    last_right_hand_size: float = 0.0

    # Smoothed ROI state: (center_x, center_y, size) for each ROI slot
    # None means no previous frame data yet
    smooth_left_hand_roi: tuple[float, float, float] | None = None
    smooth_right_hand_roi: tuple[float, float, float] | None = None
    smooth_face_roi: tuple[float, float, float] | None = None

    @classmethod
    def create(cls, config: MediapipeCompositeDetectorConfig) -> "MediapipeCompositeDetector":
        pose_detector = MediapipePoseDetector.create(config=config.pose_config)

        hand_detector: MediapipeHandDetector | None = None
        if config.detect_hands:
            hand_detector = MediapipeHandDetector.create(config=config.hand_config)

        face_detector: MediapipeFaceDetector | None = None
        if config.detect_face:
            face_detector = MediapipeFaceDetector.create(config=config.face_config)

        return cls(
            config=config,
            pose_detector=pose_detector,
            hand_detector=hand_detector,
            face_detector=face_detector,
            tracked_object=MEDIAPIPE_HOLISTIC_DEFINITION,
        )

    def detect(self, frame_number: int, image: np.ndarray) -> MediapipeCompositeObservation:
        image_h, image_w = image.shape[:2]
        image_size = (image_h, image_w)

        # Convert BGR (from OpenCV) to RGB (expected by MediaPipe)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 1: Run pose on full image
        # PoseLandmarker is in VIDEO mode which uses timestamps for temporal smoothing.
        # Pass real wall-clock milliseconds so smoothing behaves correctly.
        timestamp_ms = int(time.monotonic() * 1000)
        pose_obs = self.pose_detector.detect(frame_number=frame_number, image=rgb_image, timestamp_ms=timestamp_ms)

        # If no body detected, return empty observation (skip expensive hand/face detection)
        if not pose_obs.has_detection:
            return MediapipeCompositeObservation.build(
                frame_number=frame_number,
                image_size=image_size,
                pose=pose_obs,
                hands=None,
                face=None,
            )

        body_xyz = pose_obs.body_landmarks_xyz
        body_vis = pose_obs.body_visibility

        # Step 2: Run hand and face detection in parallel.
        # MediaPipe's C++ inference releases the GIL, so ThreadPoolExecutor
        # gives real parallelism here — all three sub-detections can run
        # simultaneously on separate threads.
        hand_obs: MediapipeHandObservation | None = None
        left_hand_roi: ROIBox | None = None
        right_hand_roi: ROIBox | None = None
        face_obs: MediapipeFaceObservation | None = None
        face_roi_box: ROIBox | None = None

        futures = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            if self.hand_detector is not None:
                futures["right_hand"] = pool.submit(
                    self._detect_hand,
                    frame_number=frame_number,
                    image=rgb_image,
                    body_xyz=body_xyz,
                    body_vis=body_vis,
                    wrist_index=POSE_RIGHT_WRIST_INDEX,
                    elbow_index=POSE_RIGHT_ELBOW_INDEX,
                    handedness_hint="Right",
                )
                futures["left_hand"] = pool.submit(
                    self._detect_hand,
                    frame_number=frame_number,
                    image=rgb_image,
                    body_xyz=body_xyz,
                    body_vis=body_vis,
                    wrist_index=POSE_LEFT_WRIST_INDEX,
                    elbow_index=POSE_LEFT_ELBOW_INDEX,
                    handedness_hint="Left",
                )
            if self.face_detector is not None:
                futures["face"] = pool.submit(
                    self._detect_face,
                    frame_number=frame_number,
                    image=rgb_image,
                    body_xyz=body_xyz,
                    body_vis=body_vis,
                )

        # Collect results
        if "right_hand" in futures:
            right_hand_obs, right_hand_roi = futures["right_hand"].result()
            left_hand_obs, left_hand_roi = futures["left_hand"].result()

            # Resolve overlapping hand detections
            right_hand_obs, left_hand_obs = self._resolve_hand_overlap(
                right_hand_obs=right_hand_obs,
                left_hand_obs=left_hand_obs,
                body_xyz=body_xyz,
            )

            # Merge left and right hand observations
            hand_obs = MediapipeHandObservation.from_arrays(
                frame_number=frame_number,
                image_size=image_size,
                right_hand_xyz=right_hand_obs.right_hand_landmarks_xyz.copy(),
                left_hand_xyz=left_hand_obs.left_hand_landmarks_xyz.copy(),
                right_hand_visibility=right_hand_obs.right_hand_visibility.copy(),
                left_hand_visibility=left_hand_obs.left_hand_visibility.copy(),
            )

        if "face" in futures:
            face_obs, face_roi_box = futures["face"].result()

        return MediapipeCompositeObservation.build(
            frame_number=frame_number,
            image_size=image_size,
            pose=pose_obs,
            hands=hand_obs,
            face=face_obs,
            left_hand_roi=left_hand_roi,
            right_hand_roi=right_hand_roi,
            face_roi=face_roi_box,
        )

    def _detect_hand(
        self,
        frame_number: int,
        image: np.ndarray,
        body_xyz: np.ndarray,
        body_vis: np.ndarray,
        wrist_index: int,
        elbow_index: int,
        handedness_hint: str,
    ) -> tuple[MediapipeHandObservation, ROIBox | None]:
        """Detect a single hand, using ROI crop if wrist is visible, otherwise full image."""
        assert self.hand_detector is not None

        image_h, image_w = image.shape[:2]
        wrist_vis = body_vis[wrist_index]

        if wrist_vis >= self.config.roi_visibility_threshold:
            wrist_xy = body_xyz[wrist_index, :2]
            elbow_xy = body_xyz[elbow_index, :2]
            arm_length = float(np.linalg.norm(wrist_xy - elbow_xy))

            # Previous frame's hand bounding box diagonal — immune to body foreshortening
            last_hand_size = (
                self.last_right_hand_size if handedness_hint == "Right"
                else self.last_left_hand_size
            )

            # Compute candidate crop sizes from different references.
            # Each uses its own appropriate multiplier — don't stack them.
            arm_crop = arm_length * self.config.hand_roi_scale
            hand_crop = last_hand_size * self.config.hand_bbox_padding if last_hand_size > 0.0 else 0.0
            min_crop = image_h * self.config.min_hand_crop_image_fraction

            crop_size = int(max(arm_crop, hand_crop, min_crop))

            # Smooth the ROI center and size against previous frame
            roi_slot = "right" if handedness_hint == "Right" else "left"
            cx, cy, sz = self._smooth_roi_params(
                raw_cx=float(wrist_xy[0]),
                raw_cy=float(wrist_xy[1]),
                raw_size=float(crop_size),
                slot=roi_slot,
            )

            roi = self._compute_square_roi(
                center_x=int(cx),
                center_y=int(cy),
                size=int(sz),
                image_w=image_w,
                image_h=image_h,
            )

            crop = image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width].copy()
            if crop.size > 0:
                obs = self.hand_detector.detect_in_crop(
                    frame_number=frame_number,
                    crop=crop,
                    crop_origin=(roi.y, roi.x),
                    full_image_size=(image_h, image_w),
                    handedness_hint=handedness_hint,
                )
                # Update the stored hand size from this detection
                self._update_hand_size(obs=obs, handedness_hint=handedness_hint)
                return obs, roi

        # Fallback: full-image detection
        obs = self.hand_detector.detect(frame_number=frame_number, image=image)
        self._update_hand_size(obs=obs, handedness_hint=handedness_hint)
        return self._filter_hand_by_side(obs=obs, handedness_hint=handedness_hint, image_size=(image_h, image_w), frame_number=frame_number), None

    def _resolve_hand_overlap(
        self,
        right_hand_obs: MediapipeHandObservation,
        left_hand_obs: MediapipeHandObservation,
        body_xyz: np.ndarray,
    ) -> tuple[MediapipeHandObservation, MediapipeHandObservation]:
        """
        Detect when both hand detections landed on the same physical hand
        (overlapping wrist positions) and resolve by assigning to the
        nearest body wrist, dropping the duplicate.
        """
        if not right_hand_obs.has_right_hand or not left_hand_obs.has_left_hand:
            return right_hand_obs, left_hand_obs

        rh_wrist = right_hand_obs.right_hand_landmarks_xyz[HAND_WRIST_INDEX, :2]
        lh_wrist = left_hand_obs.left_hand_landmarks_xyz[HAND_WRIST_INDEX, :2]

        if np.isnan(rh_wrist).any() or np.isnan(lh_wrist).any():
            return right_hand_obs, left_hand_obs

        # Check overlap: are detected hand wrists close relative to hand size?
        hand_distance = float(np.linalg.norm(rh_wrist - lh_wrist))
        reference_size = max(
            self._hand_bbox_diagonal(right_hand_obs.right_hand_landmarks_xyz),
            self._hand_bbox_diagonal(left_hand_obs.left_hand_landmarks_xyz),
        )

        if reference_size <= 0.0 or hand_distance / reference_size > self.config.hand_overlap_threshold:
            return right_hand_obs, left_hand_obs

        # Both detections found the same hand. Figure out which body wrist
        # this hand belongs to and kill the other.
        body_right_wrist = body_xyz[POSE_RIGHT_WRIST_INDEX, :2]
        body_left_wrist = body_xyz[POSE_LEFT_WRIST_INDEX, :2]

        if np.isnan(body_right_wrist).any() or np.isnan(body_left_wrist).any():
            return right_hand_obs, left_hand_obs

        # Use the mean of the two overlapping detections as the hand's position
        mean_wrist = (rh_wrist + lh_wrist) / 2.0
        dist_to_right_body = float(np.linalg.norm(mean_wrist - body_right_wrist))
        dist_to_left_body = float(np.linalg.norm(mean_wrist - body_left_wrist))

        frame_number = right_hand_obs.frame_number
        image_size = right_hand_obs.image_size
        nan_hand = np.full((NUM_HAND_LANDMARKS, 3), np.nan)
        zero_vis = np.zeros(NUM_HAND_LANDMARKS)
        empty = MediapipeHandObservation.from_arrays(
            frame_number=frame_number,
            image_size=image_size,
            right_hand_xyz=nan_hand.copy(),
            left_hand_xyz=nan_hand.copy(),
            right_hand_visibility=zero_vis.copy(),
            left_hand_visibility=zero_vis.copy(),
        )

        if dist_to_right_body <= dist_to_left_body:
            # It's the right hand — keep right detection, drop left
            return right_hand_obs, empty
        else:
            # It's the left hand — keep left detection, drop right
            return empty, left_hand_obs

    def _update_hand_size(self, obs: MediapipeHandObservation, handedness_hint: str) -> None:
        """Update stored hand size from the bounding box diagonal of detected landmarks."""
        if handedness_hint == "Right" and obs.has_right_hand:
            self.last_right_hand_size = self._hand_bbox_diagonal(obs.right_hand_landmarks_xyz)
        elif handedness_hint == "Left" and obs.has_left_hand:
            self.last_left_hand_size = self._hand_bbox_diagonal(obs.left_hand_landmarks_xyz)

    @staticmethod
    def _hand_bbox_diagonal(landmarks_xyz: np.ndarray) -> float:
        """Compute the bounding box diagonal of hand landmarks in pixel space."""
        points_2d = landmarks_xyz[:, :2]
        valid = points_2d[~np.isnan(points_2d).any(axis=1)]
        if len(valid) < 2:
            return 0.0
        mins = valid.min(axis=0)
        maxs = valid.max(axis=0)
        return float(np.linalg.norm(maxs - mins))

    def _detect_face(
        self,
        frame_number: int,
        image: np.ndarray,
        body_xyz: np.ndarray,
        body_vis: np.ndarray,
    ) -> tuple[MediapipeFaceObservation, ROIBox | None]:
        """
        Detect face using an ROI crop derived from the pose head landmarks.

        The crop is a square whose side length is face_roi_scale times the
        largest dimension of the tight bounding box over all visible head
        landmarks (nose, eyes, ears, mouth).  Using the full set of visible
        head points makes the crop robust to side-on views where one ear — or
        both ears — may be completely occluded.

        Falls back to full-image detection when fewer than 2 head landmarks
        have sufficient visibility.
        """
        assert self.face_detector is not None

        image_h, image_w = image.shape[:2]

        # Collect 2-D positions of all head landmarks that exceed the visibility threshold.
        visible_head_points: list[np.ndarray] = []
        for idx in HEAD_POSE_INDICES:
            if body_vis[idx] >= self.config.roi_visibility_threshold:
                xy = body_xyz[idx, :2]
                if not np.isnan(xy).any():
                    visible_head_points.append(xy)

        if len(visible_head_points) >= 2:
            pts = np.stack(visible_head_points, axis=0)  # shape (N, 2)
            min_xy = pts.min(axis=0)
            max_xy = pts.max(axis=0)
            bbox_center = (min_xy + max_xy) / 2.0
            bbox_w = float(max_xy[0] - min_xy[0])
            bbox_h = float(max_xy[1] - min_xy[1])

            # Square crop side = scale factor applied to the larger bbox dimension.
            # face_roi_scale = 1.5 means the crop is 50 % larger than the tight bbox.
            crop_size = int(max(bbox_w, bbox_h) * self.config.face_roi_scale)

            if crop_size > 1:
                cx, cy, sz = self._smooth_roi_params(
                    raw_cx=float(bbox_center[0]),
                    raw_cy=float(bbox_center[1]),
                    raw_size=float(crop_size),
                    slot="face",
                )

                roi = self._compute_square_roi(
                    center_x=int(cx),
                    center_y=int(cy),
                    size=int(sz),
                    image_w=image_w,
                    image_h=image_h,
                )

                crop = image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width].copy()
                if crop.size > 0:
                    obs = self.face_detector.detect_in_crop(
                        frame_number=frame_number,
                        crop=crop,
                        crop_origin=(roi.y, roi.x),
                        full_image_size=(image_h, image_w),
                    )
                    return obs, roi

        # Fallback: full-image detection when too few head landmarks are visible.
        obs = self.face_detector.detect(frame_number=frame_number, image=image)
        return obs, None

    def _smooth_roi_params(
        self,
        raw_cx: float,
        raw_cy: float,
        raw_size: float,
        slot: str,
    ) -> tuple[float, float, float]:
        """
        Apply exponential moving average smoothing to ROI center and size.

        Args:
            raw_cx: Raw center X from this frame's landmarks.
            raw_cy: Raw center Y from this frame's landmarks.
            raw_size: Raw crop size from this frame's computation.
            slot: One of "left", "right", "face" — which ROI to smooth.

        Returns:
            Smoothed (center_x, center_y, size).
        """
        alpha = self.config.roi_smoothing

        if slot == "left":
            prev = self.smooth_left_hand_roi
        elif slot == "right":
            prev = self.smooth_right_hand_roi
        elif slot == "face":
            prev = self.smooth_face_roi
        else:
            raise ValueError(f"Unknown ROI slot: {slot}")

        if prev is None:
            smoothed = (raw_cx, raw_cy, raw_size)
        else:
            prev_cx, prev_cy, prev_size = prev
            smoothed = (
                alpha * prev_cx + (1.0 - alpha) * raw_cx,
                alpha * prev_cy + (1.0 - alpha) * raw_cy,
                alpha * prev_size + (1.0 - alpha) * raw_size,
            )

        if slot == "left":
            self.smooth_left_hand_roi = smoothed
        elif slot == "right":
            self.smooth_right_hand_roi = smoothed
        elif slot == "face":
            self.smooth_face_roi = smoothed

        return smoothed

    @staticmethod
    def _compute_square_roi(center_x: int, center_y: int, size: int, image_w: int, image_h: int) -> ROIBox:
        """Compute a square ROI clamped to image bounds."""
        half = size // 2
        x = max(0, center_x - half)
        y = max(0, center_y - half)
        x2 = min(image_w, center_x + half)
        y2 = min(image_h, center_y + half)
        return ROIBox(x=x, y=y, width=x2 - x, height=y2 - y)

    @staticmethod
    def _filter_hand_by_side(
        obs: MediapipeHandObservation,
        handedness_hint: str,
        image_size: tuple[int, int],
        frame_number: int,
    ) -> MediapipeHandObservation:
        """
        From a full-image hand observation (which may have both hands),
        return an observation with only the requested side populated.
        """
        nan_hand = np.full((NUM_HAND_LANDMARKS, 3), np.nan)
        zero_vis = np.zeros(NUM_HAND_LANDMARKS)

        if handedness_hint == "Right":
            return MediapipeHandObservation.from_arrays(
                frame_number=frame_number,
                image_size=image_size,
                right_hand_xyz=obs.right_hand_landmarks_xyz.copy(),
                left_hand_xyz=nan_hand,
                right_hand_visibility=obs.right_hand_visibility.copy(),
                left_hand_visibility=zero_vis,
            )
        else:
            return MediapipeHandObservation.from_arrays(
                frame_number=frame_number,
                image_size=image_size,
                right_hand_xyz=nan_hand,
                left_hand_xyz=obs.left_hand_landmarks_xyz.copy(),
                right_hand_visibility=zero_vis,
                left_hand_visibility=obs.left_hand_visibility.copy(),
            )

    def close(self) -> None:
        self.pose_detector.close()
        if self.hand_detector is not None:
            self.hand_detector.close()
        if self.face_detector is not None:
            self.face_detector.close()
