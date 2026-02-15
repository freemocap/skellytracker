from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetectorConfig


class MediapipeFaceConfig(BaseDetectorConfig):
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    confidence_threshold: float = 0.5
    num_faces: int = 1
    output_face_blendshapes: bool = True
