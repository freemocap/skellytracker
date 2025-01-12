from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import mediapipe as mp
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker import BaseDetectorConfig, BaseDetector
from skellytracker.trackers.mediapipe_tracker.mediapipe_observation import MediapipeObservation, MediapipeResults


class MediapipeModelComplexity(int, Enum):
    LITE = 0
    FULL = 1
    HEAVY = 2


class MediapipeDetectorConfig(BaseDetectorConfig):
    model_complexity: MediapipeModelComplexity = MediapipeModelComplexity.HEAVY.value
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False
    smooth_landmarks: bool = True
    enable_segmentation: bool = True
    refine_face_landmarks: bool = True
    smooth_segmentation: bool = True


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
            enable_segmentation=config.enable_segmentation,
            refine_face_landmarks=config.refine_face_landmarks,
            smooth_segmentation=config.smooth_segmentation,

        )
        return cls(
            config=config,
            detector=detector,
        )

    def detect(self, image: np.ndarray) -> tuple[MediapipeObservation, MediapipeResults]:
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # assumes we're always getting BGR input - check with Jon to verify - JSM - BGR is an old standard, lets always convert to RGB on read and then assume RGB throughout
        mediapipe_results:MediapipeResults = self.detector.process(image)
        return MediapipeObservation.from_holistic_results(mediapipe_results,
                                                          image_size=(int(image.shape[0]), int(image.shape[1])),
                                                          ), mediapipe_results
