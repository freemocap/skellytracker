import json
import numpy as np
from pathlib import Path
from skellytracker.trackers.base_tracker.base_recorder import BaseRecorder
import re
from tqdm import tqdm

class OpenPoseRecorder(BaseRecorder):
    def __init__(self, json_directory_path):
        super().__init__()
        self.json_directory_path = Path(json_directory_path)

    def extract_frame_index(self, filename):
        """Extract the numeric part indicating the frame index from the filename."""
        match = re.search(r"_(\d{12})_keypoints", filename)
        return int(match.group(1)) if match else None

    def record(self, tracked_objects=None, annotated_image=None) -> None:
        """
        Override the record method to read from JSON files instead of receiving data directly.
        """
        # This method is adapted to fit the requirement of reading from JSON,
        # hence tracked_objects and annotated_image are not used.
        self.recorded_objects = self.parse_openpose_jsons(self.json_directory_path)

    def parse_openpose_jsons(self, main_directory):
        subdirectories = [d for d in main_directory.iterdir() if d.is_dir()]
        num_cams = len(subdirectories)

        # Assuming the first subdirectory to determine the number of frames
        sample_files = list(subdirectories[0].glob("*.json"))
        num_frames = len(sample_files)
        frame_indices = [self.extract_frame_index(f.name) for f in sample_files]
        frame_indices.sort()

        # Assuming standard OpenPose output
        body_markers, hand_markers, face_markers = 25, 21, 70
        num_markers = body_markers + 2 * hand_markers + face_markers

        data_array = np.full((num_cams, num_frames, num_markers, 3), np.nan)

        for cam_index, subdir in enumerate(subdirectories):
            json_files = sorted(subdir.glob("*.json"), key=lambda x: self.extract_frame_index(x.stem))

            for file_index, json_file in enumerate(tqdm(json_files, desc=f'Processing {subdir.name} JSONs')):
                with open(json_file) as f:
                    data = json.load(f)

                if data["people"]:
                    keypoints = self.extract_keypoints(data["people"][0], body_markers, hand_markers, face_markers)
                    data_array[cam_index, frame_indices[file_index], :, :] = keypoints

        return data_array

    def extract_keypoints(self,person_data, body_markers, hand_markers, face_markers):
        """Extract and organize keypoints from person data."""
        # Initialize a full array of NaNs for keypoints
        keypoints_array = np.full((body_markers + 2 * hand_markers + face_markers, 3), np.nan)
        
        # Populate the array with available data
        if "pose_keypoints_2d" in person_data:
            keypoints_array[:body_markers, :] = np.reshape(person_data["pose_keypoints_2d"], (-1, 3))[:body_markers, :]
        if "hand_left_keypoints_2d" in person_data and "hand_right_keypoints_2d" in person_data:
            keypoints_array[body_markers:body_markers + hand_markers, :] = np.reshape(person_data["hand_left_keypoints_2d"], (-1, 3))[:hand_markers, :]
            keypoints_array[body_markers + hand_markers:body_markers + 2*hand_markers, :] = np.reshape(person_data["hand_right_keypoints_2d"], (-1, 3))[:hand_markers, :]
        if "face_keypoints_2d" in person_data:
            keypoints_array[body_markers + 2*hand_markers:, :] = np.reshape(person_data["face_keypoints_2d"], (-1, 3))[:face_markers, :]

        return keypoints_array

    def process_tracked_objects(self, **kwargs) -> np.ndarray:
        """
        Convert the recorded JSON data into the structured numpy array format.
        """
        # In this case, the recorded_objects are already in the desired format,
        # so we simply return them.
        self.recorded_objects_array = self.recorded_objects
        return self.recorded_objects_array
