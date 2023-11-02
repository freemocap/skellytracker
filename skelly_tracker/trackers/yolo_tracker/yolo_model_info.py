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
    yolo_marker_dict = {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle",
    }
