import mediapipe as mp
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetector
from skellytracker.trackers.mediapipe_tracker.mediapipe_detector_config import (
    MediapipeDetectorConfig,
    MediapipeModelComplexity,
    MEDIAPIPE_TRACKER_REALTIME_PRESET,
    MEDIAPIPE_TRACKER_POSTHOC_PRESET,
)
from skellytracker.trackers.mediapipe_tracker.mediapipe_observation import MediapipeObservation, MediapipeResults

class MediapipeDetector(BaseDetector):
    config: MediapipeDetectorConfig
    detector: mp.solutions.holistic.Holistic

    @classmethod
    def create(cls, config: MediapipeDetectorConfig|None=None) -> "MediapipeDetector":
        if config is None:
            config = MediapipeDetectorConfig()
        detector = mp.solutions.holistic.Holistic(
            model_complexity=config.model_complexity.value,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            static_image_mode=config.static_image_mode,
            smooth_landmarks=config.smooth_landmarks,
            enable_segmentation=config.enable_segmentation,
            refine_face_landmarks=config.refine_face_landmarks,
            smooth_segmentation=config.smooth_segmentation,

        )
        return cls(
            config=config,
            detector=detector,
        )
    @classmethod
    def create_realtime_preset(cls) -> "MediapipeDetector":
        return cls.create(config=MEDIAPIPE_TRACKER_REALTIME_PRESET)

    @classmethod
    def create_posthoc_preset(cls) -> "MediapipeDetector":
        return cls.create(config=MEDIAPIPE_TRACKER_POSTHOC_PRESET)

    def detect(self, frame_number: int, image: np.ndarray) -> MediapipeObservation:
        mediapipe_results: MediapipeResults = self.detector.process(image)
        return MediapipeObservation.from_detection_results(frame_number=frame_number,
                                                          mediapipe_results=mediapipe_results,
                                                          image_size=(int(image.shape[0]), int(image.shape[1])),
                                                          include_segmentation_mask=self.config.enable_segmentation
                                                          )
