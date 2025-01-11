from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import mediapipe as mp

from skellytracker.trackers.base_tracker.base_tracker import BaseDetectorConfig, BaseDetector
from skellytracker.trackers.mediapipe_tracker.mediapipe_observation import MediapipeObservation

class MediapipeModelComplexity(int, Enum):
    LITE = 0
    FULL = 1
    HEAVY = 2

class MediapipeDetectorConfig(BaseDetectorConfig):
    model_complexity: MediapipeModelComplexity = MediapipeModelComplexity.HEAVY.value
    min_detection_confidence:float = 0.5
    min_tracking_confidence:float = 0.5
    static_image_mode:bool = False
    smooth_landmarks:bool = True


@dataclass
class MediapipeDetector(BaseDetector):
    config: MediapipeDetectorConfig
    detector: mp.solutions.holistic.Holistic

    @classmethod
    def create(cls, config: MediapipeDetectorConfig):
        detector = mp.solutions.holistic.Holistic(
            model_complexity=config.model_complexity,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            static_image_mode=config.static_image_mode,
            smooth_landmarks=config.smooth_landmarks,
        )
        return cls(
            config=config,
            detector=detector,
        )

    def detect(self, image: np.ndarray) -> MediapipeObservation:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # assumes we're always getting BGR input - check with Jon to verify
        return MediapipeObservation.from_holistic_results(self.detector.process(rgb_image),
                                                            image_size=(int(image.shape[0]), int(image.shape[1])),
                                                            )
