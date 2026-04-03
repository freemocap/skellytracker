import mediapipe as mp
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetector
from skellytracker.trackers.legacy_mediapipe_tracker.legacy_mediapipe_detector_config import (
    MediapipeDetectorConfig,
    MediapipeModelComplexity,
    MEDIAPIPE_TRACKER_REALTIME_PRESET,
    MEDIAPIPE_TRACKER_POSTHOC_PRESET,
)
from skellytracker.trackers.legacy_mediapipe_tracker.legacy_mediapipe_observation import LegacyMediapipeObservation, LegacyMediapipeResults

from skellytracker.trackers.legacy_mediapipe_tracker import LegacyMediapipeDetectorConfig


class LegacyMediapipeDetector(BaseDetector):
    config: LegacyMediapipeDetectorConfig
    detector: mp.solutions.holistic.Holistic

    @classmethod
    def create(cls, config: LegacyMediapipeDetectorConfig|None=None) -> "LegacyMediapipeDetector":
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
    def create_realtime_preset(cls) -> "LegacyMediapipeDetector":
        return cls.create(config=MEDIAPIPE_TRACKER_REALTIME_PRESET)

    @classmethod
    def create_posthoc_preset(cls) -> "LegacyMediapipeDetector":
        return cls.create(config=MEDIAPIPE_TRACKER_POSTHOC_PRESET)

    def detect(self, frame_number: int, image: np.ndarray) -> LegacyMediapipeObservation:
        mediapipe_results: LegacyMediapipeResults = self.detector.process(image)
        return LegacyMediapipeObservation.from_detection_results(frame_number=frame_number,
                                                          mediapipe_results=mediapipe_results,
                                                          image_size=(int(image.shape[0]), int(image.shape[1])),
                                                          include_segmentation_mask=self.config.enable_segmentation
                                                          )
