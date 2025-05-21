from skellytracker.trackers.base_tracker.model_info import ModelInfo

#https://github.com/open-mmlab/mmpose/blob/dev-1.x/configs/_base_/datasets/coco_wholebody.py for marker order
class RTMPoseModelInfo(ModelInfo):
    name = "rtmpose"
    tracker_name = "RTMPoseTracker"

    body_landmark_names = [
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
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
    ]

    face_landmark_names = [f"face_{i}" for i in range(68)]

    hand_landmark_names = [
        "left_hand_root", "left_thumb1", "left_thumb2", "left_thumb3", "left_thumb4",
        "left_forefinger1", "left_forefinger2", "left_forefinger3", "left_forefinger4",
        "left_middle_finger1", "left_middle_finger2", "left_middle_finger3", "left_middle_finger4",
        "left_ring_finger1", "left_ring_finger2", "left_ring_finger3", "left_ring_finger4",
        "left_pinky_finger1", "left_pinky_finger2", "left_pinky_finger3", "left_pinky_finger4",
        "right_hand_root", "right_thumb1", "right_thumb2", "right_thumb3", "right_thumb4",
        "right_forefinger1", "right_forefinger2", "right_forefinger3", "right_forefinger4",
        "right_middle_finger1", "right_middle_finger2", "right_middle_finger3", "right_middle_finger4",
        "right_ring_finger1", "right_ring_finger2", "right_ring_finger3", "right_ring_finger4",
        "right_pinky_finger1", "right_pinky_finger2", "right_pinky_finger3", "right_pinky_finger4"
    ]


    landmark_names = body_landmark_names + face_landmark_names + hand_landmark_names

    num_tracked_points_body = len(body_landmark_names)
    num_tracked_points_face = len(face_landmark_names)
    num_tracked_points_left_hand = 21  # root + 4 fingers × 5 joints
    num_tracked_points_right_hand = 21
    num_tracked_points = (
        num_tracked_points_body + num_tracked_points_face + num_tracked_points_left_hand + num_tracked_points_right_hand
    )

    tracked_object_names = [
        "pose_landmarks",
        "face_landmarks",
        "left_hand_landmarks",
        "right_hand_landmarks",

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
            "marker_names": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
            "marker_weights": [0.25, 0.25, 0.25, 0.25],
        },
        "hips_center": {
            "marker_names": ["left_hip", "right_hip"],
            "marker_weights": [0.5, 0.5],
        },
    }

    segment_connections = {
        "head": {"proximal": "left_ear", "distal": "right_ear"},
        "neck": {"proximal": "head_center", "distal": "neck_center"},
        "spine": {"proximal": "neck_center", "distal": "hips_center"},
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
        "right_foot": {"proximal": "right_ankle", "distal": "right_big_toe"},
        "left_foot": {"proximal": "left_ankle", "distal": "left_big_toe"},
        "right_heel": {"proximal": "right_ankle", "distal": "right_heel"},
        "left_heel": {"proximal": "left_ankle", "distal": "left_heel"},
        "right_foot_bottom": {"proximal": "right_heel", "distal": "right_big_toe"},
        "left_foot_bottom": {"proximal": "left_heel", "distal": "left_big_toe"},
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
            "segment_com_percentage": 0.100,
        },
        "left_thigh": {
            "segment_com_length": 0.433,
            "segment_com_percentage": 0.100,
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
        "hips_center": ["left_hip", "right_hip", "trunk_center"],
        "trunk_center": ["neck_center"],
        "neck_center": ["left_shoulder", "right_shoulder", "head_center"],
        "head_center": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
        "left_shoulder": ["left_elbow"],
        "left_elbow": ["left_wrist"],
        "right_shoulder": ["right_elbow"],
        "right_elbow": ["right_wrist"],
        "left_hip": ["left_knee"],
        "left_knee": ["left_ankle"],
        "left_ankle": ["left_big_toe", "left_heel", "left_small_toe"],
        "right_hip": ["right_knee"],
        "right_knee": ["right_ankle"],
        "right_ankle": ["right_big_toe", "right_heel", "right_small_toe"],
    }

