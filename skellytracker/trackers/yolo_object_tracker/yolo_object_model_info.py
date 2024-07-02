from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams


yolo_object_model_dictionary = {
    "nano": "yolov10n.pt",
    "small": "yolov10s.pt",
    "medium": "yolov10m.pt",
    "large": "yolov10l.pt",
    "extra_large": "yolov10x.pt",
}


class YOLOObjectTrackingParams(BaseTrackingParams):
    model_size: str = "medium"
    person_only: bool = True
    confidence_threshold: float = 0.5
