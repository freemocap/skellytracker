from enum import Enum

import mediapipe as mp
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetectorConfig, BaseDetector
from skellytracker.trackers.mediapipe_tracker.mediapipe_observation import MediapipeObservation, MediapipeResults


class MediapipeModelComplexity(int, Enum):
    LITE = 0  # BlazePose Lite model, fastest
    FULL = 1  # BlazePose Full model, balanced
    HEAVY = 2  # BlazePose Heavy model, most accurate


class MediapipeDetectorConfig(BaseDetectorConfig):
    model_complexity: MediapipeModelComplexity = MediapipeModelComplexity.HEAVY
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False
    smooth_landmarks: bool = True
    enable_segmentation: bool = True
    smooth_segmentation: bool = True
    refine_face_landmarks: bool = True # adds iris landmarks to face mesh

MEDIAPIPE_TRACKER_REALTIME_PRESET = MediapipeDetectorConfig(
    model_complexity=MediapipeModelComplexity.LITE,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=False,
    refine_face_landmarks=True,
)
MEDIAPIPE_TRACKER_POSTHOC_PRESET = MediapipeDetectorConfig(
    model_complexity=MediapipeModelComplexity.HEAVY,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False,
    smooth_landmarks=True,
    enable_segmentation=True,
    smooth_segmentation=True,
    refine_face_landmarks=True,
)

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
