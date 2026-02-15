import logging
import time

import mediapipe as mp
import numpy as np
from pydantic import ConfigDict

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetector
from skellytracker.trackers.mediapipe_tracker.mediapipe_model_manager import get_pose_model_path
from skellytracker.trackers.mediapipe_tracker.body.mediapipe_pose_config import MediapipePoseConfig
from skellytracker.trackers.mediapipe_tracker.body.mediapipe_pose_observation import MediapipePoseObservation

logger = logging.getLogger(__name__)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class MediapipePoseDetector(BaseDetector):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: MediapipePoseConfig
    landmarker: PoseLandmarker

    @classmethod
    def create(cls, config: MediapipePoseConfig) -> "MediapipePoseDetector":
        model_path = get_pose_model_path(complexity=config.model_complexity)
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=config.num_poses,
            min_pose_detection_confidence=config.min_detection_confidence,
            min_pose_presence_confidence=config.min_tracking_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_segmentation_masks=config.output_segmentation_mask,
        )
        landmarker = PoseLandmarker.create_from_options(options)
        return cls(config=config, landmarker=landmarker)

    def detect(self, frame_number: int, image: np.ndarray, timestamp_ms: int | None = None) -> MediapipePoseObservation:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)
        result = self.landmarker.detect_for_video(image=mp_image, timestamp_ms=timestamp_ms)
        return MediapipePoseObservation.from_detection_results(
            frame_number=frame_number,
            pose_landmarker_result=result,
            image_size=(image.shape[0], image.shape[1]),
        )

    def close(self) -> None:
        self.landmarker.close()