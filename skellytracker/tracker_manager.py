from enum import Enum


class TrackerNames(Enum):
    MEDIAPIPE = "mediapipe"
    YOLO = "yolo"
    OPENPOSE = "openpose"
    MEDIAPIPE_BLENDSHAPES = "mediapipe_blendshapes"
    YOLO_MEDIAPIPE_COMBO = "yolo_mediapipe_combo"
    CHARUCO = "charuco"
    BRIGHTEST_POINT = "brightest_point"