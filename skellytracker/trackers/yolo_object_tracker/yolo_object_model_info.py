from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams


yolo_object_model_dictionary = {
    "nano": "yolov11n.pt",
    "small": "yolov11s.pt",
    "medium": "yolov11m.pt",
    "large": "yolov11l.pt",
    "extra_large": "yolov11x.pt",
}


class YOLOObjectTrackingParams(BaseTrackingParams):
    model_size: str = "medium"
    person_only: bool = True
    confidence_threshold: float = 0.5
