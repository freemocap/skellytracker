from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams
from skellytracker.trackers.base_tracker.model_info import ModelInfo


class YOLOModelInfo(ModelInfo):
    name = "yolo"
    tracker_name = "YOLOPoseTracker"
    num_tracked_points = 17
    model_dictionary = {
        "nano": "yolov8n-pose.pt",
        "small": "yolov8s-pose.pt",
        "medium": "yolov8m-pose.pt",
        "large": "yolov8l-pose.pt",
        "extra_large": "yolov8x-pose.pt",
        "high_res": "yolov8x-pose-p6.pt",
    } # TODO: rename to tracker_dictionary to avoid pydantic 2 conflict?
    landmark_names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    virtual_markers_definitions = {
        "head_center": {
            "marker_names": ["left_ear", "right_ear"],
            "marker_weights": [0.5, 0.5],
        },
        "neck_center": {
            "marker_names": ["left_shoulder", "right_shoulder"],
            "marker_weights": [0.5, 0.5],
        },
        "trunk_center": {
            "marker_names": [
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
            ],
            "marker_weights": [0.25, 0.25, 0.25, 0.25],
        },
        "hips_center": {
            "marker_names": ["left_hip", "right_hip"],
            "marker_weights": [0.5, 0.5],
        },
    }
    segment_connections = {
        "head": {"proximal": "left_ear", "distal": "right_ear"},
        "neck": {
            "proximal": "head_center",
            "distal": "neck_center",
        },
        "spine": {
            "proximal": "neck_center",
            "distal": "hips_center",
        },
        "right_shoulder": {"proximal": "neck_center", "distal": "right_shoulder"},
        "left_shoulder": {"proximal": "neck_center", "distal": "left_shoulder"},
        "right_upper_arm": {"proximal": "right_shoulder", "distal": "right_elbow"},
        "left_upper_arm": {"proximal": "left_shoulder", "distal": "left_elbow"},
        "right_forearm": {"proximal": "right_elbow", "distal": "right_wrist"},
        "left_forearm": {"proximal": "left_elbow", "distal": "left_wrist"},
        "right_pelvis": {"proximal": "hips_center", "distal": "right_hip"},
        "left_pelvis": {"proximal": "hips_center", "distal": "left_hip"},
        "right_thigh": {"proximal": "right_hip", "distal": "right_knee"},
        "left_thigh": {"proximal": "left_hip", "distal": "left_knee"},
        "right_shank": {"proximal": "right_knee", "distal": "right_ankle"},
        "left_shank": {"proximal": "left_knee", "distal": "left_ankle"},
    }
    center_of_mass_definitions = {
        "head": {
            "segment_com_length": 0.5,
            "segment_com_percentage": 0.081,
        },
        "spine": {
            "segment_com_length": 0.5,
            "segment_com_percentage": 0.497,
        },
        "right_upper_arm": {
            "segment_com_length": 0.436,
            "segment_com_percentage": 0.028,
        },
        "left_upper_arm": {
            "segment_com_length": 0.436,
            "segment_com_percentage": 0.028,
        },
        "right_forearm": {
            "segment_com_length": 0.682,
            "segment_com_percentage": 0.022,
        },
        "left_forearm": {
            "segment_com_length": 0.682,
            "segment_com_percentage": 0.022,
        },
        "right_thigh": {
            "segment_com_length": 0.433,
            "segment_com_percentage": 0.1,
        },
        "left_thigh": {
            "segment_com_length": 0.433,
            "segment_com_percentage": 0.1,
        },
        "right_shank": {
            "segment_com_length": 0.606,
            "segment_com_percentage": 0.061,
        },
        "left_shank": {
            "segment_com_length": 0.606,
            "segment_com_percentage": 0.061,
        },
    }
    joint_hierarchy = {
        "hips_center": ["left_hip", "right_hip", "trunk_center"],
        "trunk_center": ["neck_center"],
        "neck_center": ["left_shoulder", "right_shoulder", "head_center"],
        "head_center": [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
        ],
        "left_shoulder": ["left_elbow"],
        "left_elbow": ["left_wrist"],
        "right_shoulder": ["right_elbow"],
        "right_elbow": ["right_wrist"],
        "left_hip": ["left_knee"],
        "left_knee": ["left_ankle"],
        "right_hip": ["right_knee"],
        "right_knee": ["right_ankle"],
    }


class YOLOTrackingParams(BaseTrackingParams):
    model_size: str = "medium" # TODO: rename to tracker_model_size to avoid pydantic 2 conflict?
