from typing import List, Literal
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions.face_mesh import FACEMESH_NUM_LANDMARKS_WITH_IRISES

from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams
from skellytracker.trackers.base_tracker.model_info import ModelInfo


# values for segment weight and segment mass percentages taken from Winter anthropometry tables
# https://imgur.com/a/aD74j
# Winter, D.A. (2005) Biomechanics and Motor Control of Human Movement. 3rd Edition, John Wiley & Sons, Inc., Hoboken.
class MediapipeModelInfo(ModelInfo):
    name = "mediapipe"
    tracker_name = "MediapipeHolisticTracker"
    body_landmark_names = [
        landmark.name.lower() for landmark in mp_holistic.PoseLandmark
    ]
    hand_landmark_names = [
        landmark.name.lower() for landmark in mp_holistic.HandLandmark
    ]
    face_landmark_names = [
        "right_eye",
        "left_eye",
        "nose_tip",
        "mouth_center",
        "right_ear_tragion",
        "left_ear_tragion",
    ]
    landmark_names = body_landmark_names + hand_landmark_names + face_landmark_names
    num_tracked_points_body = len(body_landmark_names)
    num_tracked_points_face = FACEMESH_NUM_LANDMARKS_WITH_IRISES
    num_tracked_points_left_hand = len(hand_landmark_names)
    num_tracked_points_right_hand = len(hand_landmark_names)
    num_tracked_points = (
        len(body_landmark_names)
        + 2 * len(hand_landmark_names)
        + num_tracked_points_face
    )
    tracked_object_names = [
        "pose_landmarks",
        "right_hand_landmarks",
        "left_hand_landmarks",
        "face_landmarks",
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
        "right_hand": {"proximal": "right_wrist", "distal": "right_index"},
        "left_hand": {"proximal": "left_wrist", "distal": "left_index"},
        "right_pelvis": {"proximal": "hips_center", "distal": "right_hip"},
        "left_pelvis": {"proximal": "hips_center", "distal": "left_hip"},
        "right_thigh": {"proximal": "right_hip", "distal": "right_knee"},
        "left_thigh": {"proximal": "left_hip", "distal": "left_knee"},
        "right_shank": {"proximal": "right_knee", "distal": "right_ankle"},
        "left_shank": {"proximal": "left_knee", "distal": "left_ankle"},
        "right_foot": {"proximal": "right_ankle", "distal": "right_foot_index"},
        "left_foot": {"proximal": "left_ankle", "distal": "left_foot_index"},
        "right_heel": {"proximal": "right_ankle", "distal": "right_heel"},
        "left_heel": {"proximal": "left_ankle", "distal": "left_heel"},
        "right_foot_bottom": {"proximal": "right_heel", "distal": "right_foot_index"},
        "left_foot_bottom": {"proximal": "left_heel", "distal": "left_foot_index"},
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
            "segment_com_length": 0.430,
            "segment_com_percentage": 0.016,
        },
        "left_forearm": {
            "segment_com_length": 0.430,
            "segment_com_percentage": 0.016,
        },
        "right_hand": {
            "segment_com_length": 0.506,
            "segment_com_percentage": 0.006,
        },
        "left_hand": {
            "segment_com_length": 0.506,
            "segment_com_percentage": 0.006,
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
        "hips_center": ["left_hip", "right_hip", "trunk_center"],
        "trunk_center": ["neck_center"],
        "neck_center": ["left_shoulder", "right_shoulder", "head_center"],
        "head_center": [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
        ],
        "left_shoulder": ["left_elbow"],
        "left_elbow": ["left_wrist"],
        "left_wrist": ["left_pinky", "left_index", "left_thumb"],
        "right_shoulder": ["right_elbow"],
        "right_elbow": ["right_wrist"],
        "right_wrist": ["right_pinky", "right_index", "right_thumb"],
        "left_hip": ["left_knee"],
        "left_knee": ["left_ankle"],
        "left_ankle": ["left_heel", "left_foot_index"],
        "right_hip": ["right_knee"],
        "right_knee": ["right_ankle"],
        "right_ankle": ["right_heel", "right_foot_index"],
    }


class MediapipeTrackingParams(BaseTrackingParams):
    use_yolo_crop_method: bool = False
    mediapipe_model_complexity: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = True
    yolo_model_size: Literal[
        "nano", "small", "medium", "large", "extra_large", "high_res"
    ] = "nano"
    bounding_box_buffer_percentage: float = 10
    buffer_size_method: Literal["buffer_by_box_size", "buffer_by_image_size"] = (
        "buffer_by_box_size"
    )


def mediapipe_body_names_match_expected(
    mediapipe_body_landmark_names: List[str],
) -> bool:
    """
    Check if the mediapipe folks have changed their landmark names. If they have, then this function may need to be updated.

    Args:
        mediapipe_body_landmark_names: List of strings, each string is the name of a mediapipe landmark.

    Returns:
        bool: True if the mediapipe landmark names are as expected, False otherwise.
    """
    expected_mediapipe_body_landmark_names = [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ]
    return mediapipe_body_landmark_names == expected_mediapipe_body_landmark_names
