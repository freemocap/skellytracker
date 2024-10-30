from dataclasses import dataclass
from pathlib import Path
import requests
from typing import Literal

from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams
from skellytracker.trackers.base_tracker.model_info import ModelInfo


@dataclass
class MediapipeLandmarker:
    name: str
    model_file_name: str
    download_link: str

    def download_model(self) -> Path:
        if not Path(self.model_file_name).is_file():
            try:
                response = requests.get(self.download_link, timeout=10)
                response.raise_for_status()
                with open(self.model_file_name, "wb") as f:
                    f.write(response.content)
            except requests.exceptions.RequestException as e:
                print(f"Unable to download model from {self.download_link}: {e}")
                raise
            except IOError as e:
                print(f"Unable to save model file to {self.model_file_name}: {e}")
                raise

        print(f"Model downloaded to {self.model_file_name}")
        return Path(self.model_file_name)


# values for segment weight and segment mass percentages taken from Winter anthropometry tables
# https://imgur.com/a/aD74j
# Winter, D.A. (2005) Biomechanics and Motor Control of Human Movement. 3rd Edition, John Wiley & Sons, Inc., Hoboken.
class MediapipePoseModelInfo(ModelInfo):
    name = "mediapipe_pose"
    tracker_name = "MediapipePoseTracker"
    lite_model = MediapipeLandmarker(
        name="lite",
        model_file_name="pose_landmarker_lite.task",
        download_link="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    )
    full_model = MediapipeLandmarker(
        name="full",
        model_file_name="pose_landmarker_full.task",
        download_link="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    )
    heavy_model = MediapipeLandmarker(
        name="heavy",
        model_file_name="pose_landmarker_heavy.task",
        download_link="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    )
    landmark_names = [
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
    num_tracked_points = len(landmark_names)
    tracked_object_names = [
        "pose_landmarks",
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


class MediapipePoseTrackingParams(BaseTrackingParams):
    use_yolo_crop_method: bool = False
    model: MediapipeLandmarker = MediapipePoseModelInfo.heavy_model
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    running_mode: str = "IMAGE"
    num_poses: int = 1
    yolo_model_size: Literal[
        "nano", "small", "medium", "large", "extra_large", "high_res"
    ] = "nano"
    bounding_box_buffer_percentage: float = 10
    buffer_size_method: Literal["buffer_by_box_size", "buffer_by_image_size"] = (
        "buffer_by_box_size"
    )
