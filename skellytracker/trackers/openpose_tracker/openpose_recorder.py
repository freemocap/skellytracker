import json
from typing import Dict, Union
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm

from skellytracker.trackers.base_tracker.base_recorder import BaseCumulativeRecorder
from skellytracker.trackers.openpose_tracker.openpose_model_info import (
    OpenPoseModelInfo,
)


class OpenPoseRecorder(BaseCumulativeRecorder):
    def __init__(
        self,
        track_hands: bool = False,
        track_faces: bool = False,
    ):
        super().__init__()
        self.track_hands = track_hands
        self.track_faces = track_faces

    def extract_frame_index(self, filename: str) -> Union[int, None]:
        """Extract the numeric part indicating the frame index from the filename."""
        match = re.search(r"_(\d{12})_keypoints", filename)
        return int(match.group(1)) if match else None

    def parse_openpose_jsons(self, json_directory: Union[Path, str]) -> np.ndarray:
        # Remove the iteration over subdirectories and focus on a single directory
        json_directory = Path(json_directory)
        files = list(Path(json_directory).glob("*.json"))
        num_frames = len(files)
        frame_indices = [
            index
            for f in files
            if (index := self.extract_frame_index(f.name)) is not None
        ]
        frame_indices.sort()

        if len(frame_indices) != num_frames:
            raise ValueError(
                f"Invalid number of frames in {json_directory}: expected {num_frames} != {len(frame_indices)} frames in file"
            )

        num_markers = OpenPoseModelInfo.num_tracked_points_body
        if self.track_hands:
            num_markers += (
                OpenPoseModelInfo.num_tracked_points_right_hand
                + OpenPoseModelInfo.num_tracked_points_left_hand
            )
        if self.track_faces:
            num_markers += OpenPoseModelInfo.num_tracked_points_face

        # Initialize a single camera array since we're only processing one video at a time
        data_array = np.full((num_frames, num_markers, 3), np.nan)

        # Process each JSON file in the directory
        for file_index, json_file in enumerate(
            tqdm(files, desc=f"Processing {json_directory.name} JSONs")
        ):
            with open(json_file) as f:
                data = json.load(f)

            if data["people"]:
                keypoints = self.extract_keypoints(data["people"][0])
                data_array[frame_indices[file_index], :, :] = keypoints

        return data_array

    def extract_keypoints(self, person_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract and organize keypoints from person data."""

        body_markers = OpenPoseModelInfo.num_tracked_points_body
        hand_markers = (
            OpenPoseModelInfo.num_tracked_points_left_hand
            + OpenPoseModelInfo.num_tracked_points_right_hand
        )
        face_markers = OpenPoseModelInfo.num_tracked_points_face

        # Initialize a full array of NaNs for keypoints
        keypoints_array = np.full(
            (body_markers + (hand_markers) + face_markers, 3), np.nan
        )

        # Populate the array with available data
        if "pose_keypoints_2d" in person_data:
            keypoints_array[:body_markers, :] = np.reshape(
                person_data["pose_keypoints_2d"], (-1, 3)
            )[:body_markers, :]
        if (
            "hand_left_keypoints_2d" in person_data
            and "hand_right_keypoints_2d" in person_data
        ):
            keypoints_array[body_markers : body_markers + OpenPoseModelInfo.num_tracked_points_left_hand, :] = np.reshape(
                person_data["hand_left_keypoints_2d"], (-1, 3)
            )[:hand_markers, :]
            keypoints_array[
                body_markers + OpenPoseModelInfo.num_tracked_points_left_hand : body_markers + OpenPoseModelInfo.num_tracked_points_left_hand + OpenPoseModelInfo.num_tracked_points_right_hand, :
            ] = np.reshape(person_data["hand_right_keypoints_2d"], (-1, 3))[
                :hand_markers, :
            ]
        if "face_keypoints_2d" in person_data:
            keypoints_array[body_markers + hand_markers :, :] = np.reshape(
                person_data["face_keypoints_2d"], (-1, 3)
            )[:face_markers, :]

        return keypoints_array

    def process_tracked_objects(self, output_json_path: Path) -> np.ndarray:
        """
        Convert the recorded JSON data into the structured numpy array format.
        """
        # In this case, the recorded_objects are already in the desired format, so we simply return them.
        self.recorded_objects_array = self.parse_openpose_jsons(output_json_path)
        return self.recorded_objects_array
