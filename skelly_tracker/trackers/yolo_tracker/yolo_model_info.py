from enum import Enum

class YOLOModelInfo(Enum):
    num_tracked_points = 17
    yolo_model_dictionary = {
        "nano": "yolov8n-pose.pt",
        "small": "yolov8s-pose.pt",
        "medium": "yolov8m-pose.pt",
        "large": "yolov8l-pose.pt",
        "extra_large": "yolov8x-pose.pt",
        "high_res": "yolov8x-pose-p6.pt",
    }
