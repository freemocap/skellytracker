from enum import Enum

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetectorConfig


class MediapipeModelComplexity(int, Enum):
    LITE = 0  # BlazePose Lite model, fastest
    FULL = 1  # BlazePose Full model, balanced
    HEAVY = 2  # BlazePose Heavy model, most accurate


class MediapipeDetectorConfig(BaseDetectorConfig):
    model_complexity: MediapipeModelComplexity = MediapipeModelComplexity.HEAVY
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    confidence_threshold: float = 0.5
    static_image_mode: bool = False
    smooth_landmarks: bool = True
    enable_segmentation: bool = True
    smooth_segmentation: bool = True
    refine_face_landmarks: bool = True  # adds iris landmarks to face mesh


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
