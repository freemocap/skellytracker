from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams


yolo_object_model_dictionary = {
    "nano": "yolo11n.pt",
    "small": "yolo11s.pt",
    "medium": "yolo11m.pt",
    "large": "yolo11l.pt",
    "extra_large": "yolo11x.pt",
}


class YOLOObjectTrackingParams(BaseTrackingParams):
    model_size: str = "medium"
    person_only: bool = True
    confidence_threshold: float = 0.5
