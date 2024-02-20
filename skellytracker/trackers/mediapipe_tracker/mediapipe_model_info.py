from typing import List
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions.face_mesh import FACEMESH_NUM_LANDMARKS_WITH_IRISES

from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams
from skellytracker.trackers.base_tracker.model_info import ModelInfo


# values for segment weight and segment mass percentages taken from Winter anthropometry tables
# https://imgur.com/a/aD74j
# Winter, D.A. (2005) Biomechanics and Motor Control of Human Movement. 3rd Edition, John Wiley & Sons, Inc., Hoboken.
class MediapipeModelInfo(ModelInfo):
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
    body_connections = [connection for connection in mp_holistic.POSE_CONNECTIONS]
    hand_connections = [connection for connection in mp_holistic.HAND_CONNECTIONS]
    face_connections = [connection for connection in mp_holistic.FACEMESH_CONTOURS]
    connections = body_connections + hand_connections + face_connections
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
        "face_landmarks",
        "left_hand_landmarks",
        "right_hand_landmarks",
    ]
    segment_names = [
        "head",
        "trunk",
        "right_upper_arm",
        "left_upper_arm",
        "right_forearm",
        "left_forearm",
        "right_hand",
        "left_hand",
        "right_thigh",
        "left_thigh",
        "right_shin",
        "left_shin",
        "right_foot",
        "left_foot",
    ]
    joint_connections = [  # TODO: seems like we probably have duplications on the connections here
        ["left_ear", "right_ear"],
        ["mid_chest_marker", "mid_hip_marker"],
        ["right_shoulder", "right_elbow"],
        ["left_shoulder", "left_elbow"],
        ["right_elbow", "right_wrist"],
        ["left_elbow", "left_wrist"],
        ["right_wrist", "right_hand_marker"],
        ["left_wrist", "left_hand_marker"],
        ["right_hip", "right_knee"],
        ["left_hip", "left_knee"],
        ["right_knee", "right_ankle"],
        ["left_knee", "left_ankle"],
        ["right_back_of_foot_marker", "right_foot_index"],
        ["left_back_of_foot_marker", "left_foot_index"],
    ]
    segment_COM_lengths = [
        0.5,
        0.5,
        0.436,
        0.436,
        0.430,
        0.430,
        0.506,
        0.506,
        0.433,
        0.433,
        0.433,
        0.433,
        0.5,
        0.5,
    ]
    segment_COM_percentages = [
        0.081,
        0.497,
        0.028,
        0.028,
        0.016,
        0.016,
        0.006,
        0.006,
        0.1,
        0.1,
        0.0465,
        0.0465,
        0.0145,
        0.0145,
    ]
    names_and_connections_dict = {
        "body": {
            "names": body_landmark_names,
            "connections": body_connections,
        },
        "right_hand": {
            "names": [f"right_hand_{name}" for name in hand_landmark_names],
            "connections": hand_connections,
        },
        "left_hand": {
            "names": [f"left_hand_{name}" for name in hand_landmark_names],
            "connections": hand_connections,
        },
        "face": {
            "names": face_landmark_names,
            "connections": face_connections,
        },
    }
    virtual_marker_definitions_dict = {
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
    skeleton_schema = {
        "body": {
            "point_names": body_landmark_names,
            "connections": body_connections,
            "virtual_marker_definitions": virtual_marker_definitions_dict,
            "parent": "hips_center",
        },
        "hands": {
            "right": {
                "point_names": [name for name in hand_landmark_names],
                "connections": hand_connections,
                "parent": "right_wrist",
            },
            "left": {
                "point_names": [name for name in hand_landmark_names],
                "connections": hand_connections,
                "parent": "left_wrist",
            },
        },
        "face": {
            "point_names": face_landmark_names,
            "connections": face_connections,
            "parent": "nose",
        },
    }
    joint_hierarchy = {
        "hips_center": {"children": ["right_hip", "left_hip", "trunk_center"]},
        "trunk_center": {"children": ["neck_center"]},
        "neck_center": {"children": ["right_shoulder", "left_shoulder", "head_center"]},
        "head_center": {
            "children": [
                "nose",
                "mouth_right",
                "mouth_left",
                "right_eye",
                "right_eye_inner",
                "right_eye_outer",
                "left_eye",
                "left_eye_inner",
                "left_eye_outer",
                "right_ear",
                "left_ear",
            ]
        },
        "right_hip": {"children": ["right_knee"]},
        "right_knee": {"children": ["right_ankle"]},
        "right_ankle": {"children": ["right_foot_index", "right_heel"]},
        "left_hip": {"children": ["left_knee"]},
        "left_knee": {"children": ["left_ankle"]},
        "left_ankle": {"children": ["left_foot_index", "left_heel"]},
        "right_shoulder": {"children": ["right_elbow"]},
        "right_elbow": {"children": ["right_wrist"]},
        "right_wrist": {
            "children": [
                "right_thumb",
                "right_index",
                "right_pinky",
                "right_hand_middle",
                "right_hand_wrist",
            ]
        },
        "right_hand_wrist": {
            "children": [
                "right_hand_thumb_cmc",
                "right_hand_index_finger_mcp",
                "right_hand_middle_finger_mcp",
                "right_hand_ring_finger_mcp",
                "right_hand_pinky_mcp",
            ]
        },
        "right_hand_thumb_cmc": {"children": ["right_hand_thumb_mcp"]},
        "left_shoulder": {"children": ["left_elbow"]},
        "left_elbow": {"children": ["left_wrist"]},
        "left_wrist": {
            "children": [
                "left_thumb",
                "left_index",
                "left_pinky",
                "left_hand_middle",
                "left_hand_wrist",
            ]
        },
        "left_hand_wrist": {
            "children": [
                "left_hand_thumb_cmc",
                "left_hand_index_finger_mcp",
                "left_hand_middle_finger_mcp",
                "left_hand_ring_finger_mcp",
                "left_hand_pinky_mcp",
            ]
        },
        "right_hand_thumb_mcp": {"children": ["right_hand_thumb_ip"]},
        "right_hand_thumb_ip": {"children": ["right_hand_thumb_tip"]},
        "right_hand_index_finger_mcp": {"children": ["right_hand_index_finger_pip"]},
        "right_hand_index_finger_pip": {"children": ["right_hand_index_finger_dip"]},
        "right_hand_index_finger_dip": {"children": ["right_hand_index_finger_tip"]},
        "left_hand_thumb_cmc": {"children": ["left_hand_thumb_mcp"]},
        "right_hand_middle_finger_mcp": {"children": ["right_hand_middle_finger_pip"]},
        "right_hand_middle_finger_pip": {"children": ["right_hand_middle_finger_dip"]},
        "right_hand_middle_finger_dip": {"children": ["right_hand_middle_finger_tip"]},
        "right_hand_ring_finger_mcp": {"children": ["right_hand_ring_finger_pip"]},
        "right_hand_ring_finger_pip": {"children": ["right_hand_ring_finger_dip"]},
        "left_hand_thumb_mcp": {"children": ["left_hand_thumb_ip"]},
        "right_hand_ring_finger_dip": {"children": ["right_hand_ring_finger_tip"]},
        "left_hand_thumb_ip": {"children": ["left_hand_thumb_tip"]},
        "right_hand_pinky_mcp": {"children": ["right_hand_pinky_pip"]},
        "left_hand_index_finger_mcp": {"children": ["left_hand_index_finger_pip"]},
        "right_hand_pinky_pip": {"children": ["right_hand_pinky_dip"]},
        "right_hand_pinky_dip": {"children": ["right_hand_pinky_tip"]},
        "left_hand_index_finger_pip": {"children": ["left_hand_index_finger_dip"]},
        "left_hand_index_finger_dip": {"children": ["left_hand_index_finger_tip"]},
        "left_hand_middle_finger_mcp": {"children": ["left_hand_middle_finger_pip"]},
        "left_hand_middle_finger_pip": {"children": ["left_hand_middle_finger_dip"]},
        "left_hand_middle_finger_dip": {"children": ["left_hand_middle_finger_tip"]},
        "left_hand_ring_finger_mcp": {"children": ["left_hand_ring_finger_pip"]},
        "left_hand_ring_finger_pip": {"children": ["left_hand_ring_finger_dip"]},
        "left_hand_ring_finger_dip": {"children": ["left_hand_ring_finger_tip"]},
        "left_hand_pinky_mcp": {"children": ["left_hand_pinky_pip"]},
        "left_hand_pinky_pip": {"children": ["left_hand_pinky_dip"]},
        "left_hand_pinky_dip": {"children": ["left_hand_pinky_tip"]},
    }


class MediapipeTrackingParams(BaseTrackingParams):
    use_yolo_crop_method: bool = False
    mediapipe_model_complexity: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = True


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
