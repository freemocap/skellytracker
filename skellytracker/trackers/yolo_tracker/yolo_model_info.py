from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams


class YOLOModelInfo:
    num_tracked_points = 17
    model_dictionary = {
        "nano": "yolov8n-pose.pt",
        "small": "yolov8s-pose.pt",
        "medium": "yolov8m-pose.pt",
        "large": "yolov8l-pose.pt",
        "extra_large": "yolov8x-pose.pt",
        "high_res": "yolov8x-pose-p6.pt",
    }
    marker_dict = {
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
    body_segment_names = [
        "head",
        "trunk",
        "right_upper_arm",
        "left_upper_arm",
        "right_forearm",
        "left_forearm",
        "right_thigh",
        "left_thigh",
        "right_shin",
        "left_shin",
    ]
    joint_connections = [
        ["left_ear", "right_ear"],
        ["mid_chest_marker", "mid_hip_marker"],
        ["right_shoulder", "right_elbow"],
        ["left_shoulder", "left_elbow"],
        ["right_elbow", "right_wrist"],
        ["left_elbow", "left_wrist"],
        ["right_hip", "right_knee"],
        ["left_hip", "left_knee"],
        ["right_knee", "right_ankle"],
        ["left_knee", "left_ankle"],
    ]
    segment_COM_lengths = [
        0.5,
        0.5,
        0.436,
        0.436,
        0.682,
        0.682,
        0.433,
        0.433,
        0.606,
        0.606,
    ]
    segment_COM_percentages = [
        0.081,
        0.497,
        0.028,
        0.028,
        0.022,
        0.022,
        0.1,
        0.1,
        0.061,
        0.061,
    ]


class YOLOTrackingParams(BaseTrackingParams):
    model_size: str = "medium"
