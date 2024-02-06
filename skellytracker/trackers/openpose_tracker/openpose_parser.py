import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

def extract_frame_index(filename):
    """Extract the numeric part indicating the frame index from the filename."""
    match = re.search(r"_(\d{12})_keypoints", filename)
    return int(match.group(1)) if match else None

def parse_openpose_jsons(main_directory):
    """Parse OpenPose JSON files from subdirectories within the main directory."""
    main_directory = Path(main_directory)
    subdirectories = [d for d in main_directory.iterdir() if d.is_dir()]
    num_cams = len(subdirectories)
    
    # Check the first subdirectory to determine the number of frames
    sample_files = list(subdirectories[0].glob("*.json"))
    num_frames = len(sample_files)
    frame_indices = [extract_frame_index(f.name) for f in sample_files]
    frame_indices.sort()
    
    # Assuming standard OpenPose output
    body_markers = 25
    hand_markers = 21  # Per hand
    face_markers = 70
    num_markers = body_markers + 2 * hand_markers + face_markers

    data_array = np.full((num_cams, num_frames, num_markers, 3), np.nan)
    
    for cam_index, subdir in enumerate(subdirectories):
        json_files = sorted(subdir.glob("*.json"), key=lambda x: extract_frame_index(x.stem))
        
        for file_index, json_file in tqdm(enumerate(json_files), desc = f'Processing {subdir.name} JSONS'):
            with open(json_file) as f:
                data = json.load(f)
            
            if data["people"]:
                keypoints = extract_keypoints(data["people"][0], body_markers, hand_markers, face_markers)
                data_array[cam_index, frame_indices[file_index], :, :] = keypoints

    return data_array

def extract_keypoints(person_data, body_markers, hand_markers, face_markers):
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


path_to_recording_folder = Path(r'D:\steen_pantsOn_gait_3_cameras')
path_to_json_folder = path_to_recording_folder/'output_data'/'raw_data'/'openpose_json'
path_to_save_raw_data = path_to_recording_folder/'output_data'/'raw_data'/'openpose2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy'

data = parse_openpose_jsons(path_to_json_folder)
np.save(path_to_save_raw_data, data)
f = 2