from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetectorConfig, TrackerType
from skellytracker.trackers.mediapipe_tracker.mediapipe_model_manager import MediapipePoseModelComplexity


class MediapipePoseConfig(BaseDetectorConfig):
    tracker_type: TrackerType = TrackerType.MEDIAPIPE_POSE
    model_complexity: MediapipePoseModelComplexity = MediapipePoseModelComplexity.HEAVY
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    confidence_threshold: float = 0.5
    output_segmentation_mask: bool = True
    num_poses: int = 1
