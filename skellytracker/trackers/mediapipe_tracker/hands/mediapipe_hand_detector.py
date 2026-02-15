import logging

import mediapipe as mp
import numpy as np
from pydantic import ConfigDict

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetector
from skellytracker.trackers.mediapipe_tracker.hands.mediapipe_hand_config import MediapipeHandConfig
from skellytracker.trackers.mediapipe_tracker.hands.mediapipe_hand_observation import MediapipeHandObservation
from skellytracker.trackers.mediapipe_tracker.mediapipe_model_manager import get_hand_model_path

logger = logging.getLogger(__name__)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class MediapipeHandDetector(BaseDetector):
    """Wraps MediaPipe HandLandmarker in IMAGE mode (for use with crops)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: MediapipeHandConfig
    landmarker: HandLandmarker

    @classmethod
    def create(cls, config: MediapipeHandConfig) -> "MediapipeHandDetector":
        model_path = get_hand_model_path()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=config.num_hands,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_tracking_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        landmarker = HandLandmarker.create_from_options(options)
        return cls(config=config, landmarker=landmarker)

    def detect(self, frame_number: int, image: np.ndarray) -> MediapipeHandObservation:
        """Detect hands in a full image."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self.landmarker.detect(image=mp_image)
        return MediapipeHandObservation.from_detection_results(
            frame_number=frame_number,
            hand_landmarker_result=result,
            image_size=(image.shape[0], image.shape[1]),
        )

    def detect_in_crop(
        self,
        frame_number: int,
        crop: np.ndarray,
        crop_origin: tuple[int, int],
        full_image_size: tuple[int, int],
        handedness_hint: str,
    ) -> MediapipeHandObservation:
        """
        Detect a hand in a cropped image region and map coordinates back to full image space.

        Args:
            crop: The cropped image region (RGB).
            crop_origin: (y_offset, x_offset) of crop in the full image.
            full_image_size: (height, width) of the full image.
            handedness_hint: "Left" or "Right" — which hand to expect.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
        result = self.landmarker.detect(image=mp_image)
        return MediapipeHandObservation.from_crop_results(
            frame_number=frame_number,
            hand_landmarker_result=result,
            crop_origin=crop_origin,
            crop_size=(crop.shape[0], crop.shape[1]),
            full_image_size=full_image_size,
            handedness_hint=handedness_hint,
        )

    def close(self) -> None:
        self.landmarker.close()