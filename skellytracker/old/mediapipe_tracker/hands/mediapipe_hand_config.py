from skellytracker.old.base_tracker.base_tracker_abcs import BaseDetectorConfig, TrackerType


class MediapipeHandConfig(BaseDetectorConfig):
    tracker_type: TrackerType = TrackerType.MEDIAPIPE_HAND
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    confidence_threshold: float = 0.5
    num_hands: int = 2
