from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams
from skellytracker.trackers.base_tracker.model_info import ModelInfo

from typing import Optional


class OpenPoseModelInfo(ModelInfo):
    name = "openpose"
    tracker_name = "OpenPoseTracker"
    body_landmark_names = [
        "nose",
        "neck",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "hip_center",
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
        "right_eye",
        "left_eye", 
        "right_ear", 
        "left_ear", 
        "left_big_toe",
        "left_small_toe", 
        "left_heel", 
        "right_big_toe", 
        "right_small_toe", 
        "right_heel"
    ]
    landmark_names = body_landmark_names
    num_tracked_points_body = len(body_landmark_names)
    num_tracked_points_face = 70
    num_tracked_points_left_hand = 21
    num_tracked_points_right_hand = 21

    num_tracked_points = (
        num_tracked_points_body
        + num_tracked_points_left_hand
        + num_tracked_points_right_hand
        + num_tracked_points_face
    )
    tracked_object_names = ["pose_landmarks"]
    virtual_markers_definitions = {
        "head_center": {
            "marker_names": ["left_ear", "right_ear"],
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
    }
    segment_connections = {
        "head": {"proximal": "left_ear", "distal": "right_ear"},
        "neck": {"proximal": "head_center", "distal": "neck"},
        "spine": {"proximal": "neck", "distal": "hip_center"},
        "right_shoulder": {"proximal": "neck", "distal": "right_shoulder"},
        "left_shoulder": {"proximal": "neck", "distal": "left_shoulder"},
        "right_upper_arm": {"proximal": "right_shoulder", "distal": "right_elbow"},
        "left_upper_arm": {"proximal": "left_shoulder", "distal": "left_elbow"},
        "right_forearm": {"proximal": "right_elbow", "distal": "right_wrist"},
        "left_forearm": {"proximal": "left_elbow", "distal": "left_wrist"},
        "right_pelvis": {"proximal": "hip_center", "distal": "right_hip"},
        "left_pelvis": {"proximal": "hip_center", "distal": "left_hip"},
        "right_thigh": {"proximal": "right_hip", "distal": "right_knee"},
        "left_thigh": {"proximal": "left_hip", "distal": "left_knee"},
        "right_shank": {"proximal": "right_knee", "distal": "right_ankle"},
        "left_shank": {"proximal": "left_knee", "distal": "left_ankle"},
        "right_foot": {"proximal": "right_ankle", "distal": "right_big_toe"},
        "left_foot": {"proximal": "left_ankle", "distal": "left_big_toe"},
        "right_heel": {"proximal": "right_ankle", "distal": "right_heel"},
        "left_heel": {"proximal": "left_ankle", "distal": "left_heel"},
        "right_foot_bottom": {"proximal": "right_heel", "distal": "right_big_toe"},
        "left_foot_bottom": {"proximal": "left_heel", "distal": "left_big_toe"},
    }
    center_of_mass_definitions = { #NOTE: using forearm/hand definition from Winter tables, as we don't have hand definitions here  
        "head": {
            "segment_com_length": .5,
            "segment_com_percentage": .081,
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
            "segment_com_length": 0.433,
            "segment_com_percentage": 0.0465,
        },
        "left_shank": {
            "segment_com_length": 0.433,
            "segment_com_percentage": 0.0465,
        },
        "right_foot": {
            "segment_com_length": 0.5,
            "segment_com_percentage": 0.0145,
        },
        "left_foot": {
            "segment_com_length": 0.5,
            "segment_com_percentage": 0.0145,
        },
    }        
    joint_hierarchy = {
        "hip_center": ["left_hip", "right_hip", "trunk_center"],
        "trunk_center": ["neck"],
        "neck": ["left_shoulder", "right_shoulder", "head_center"],
        "head_center": ["nose", "left_ear", "right_ear", "left_eye", "right_eye"],
        "left_shoulder": ["left_elbow"],
        "left_elbow": ["left_wrist"],
        "right_shoulder": ["right_elbow"],
        "right_elbow": ["right_wrist"],
        "left_hip": ["left_knee"],
        "left_knee": ["left_ankle"],
        "left_ankle": ["left_big_toe", "left_small_toe", "left_heel"],
        "right_hip": ["right_knee"],
        "right_knee": ["right_ankle"],
        "right_ankle": ["right_big_toe", "right_small_toe", "right_heel"],
    }


class OpenPoseTrackingParams(BaseTrackingParams):
    openpose_root_folder_path: str
    output_json_path: Optional[str] = None
    net_resolution: str = "-1x320"
    number_people_max: int = 1
    track_hands: bool = True
    track_face: bool = True
    write_video: bool = True
    output_resolution: str = "-1x-1"
