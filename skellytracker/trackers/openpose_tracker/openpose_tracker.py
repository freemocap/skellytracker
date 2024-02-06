import subprocess
from pathlib import Path
from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.openpose_tracker.openpose_recorder import OpenPoseRecorder
from typing import Dict, Any
import numpy as np

import os
class OpenPoseTracker(BaseTracker):
    def __init__(self, openpose_exe_path, output_json_path, net_resolution="-1x640", number_people_max=1):
        """
        Initialize the OpenPoseTracker.

        :param recorder: An instance of OpenPoseRecorder for handling the output.
        :param openpose_exe_path: Path to the OpenPose executable.
        :param output_json_path: Directory where JSON files will be saved.
        :param net_resolution: Network resolution for OpenPose processing.
        :param number_people_max: Maximum number of people to detect.
        """
        # super().__init__(recorder=recorder)
        self.openpose_exe_path = Path(openpose_exe_path)
        self.output_json_path = Path(output_json_path)
        self.net_resolution = net_resolution
        self.number_people_max = number_people_max

        super().__init__(
            tracked_object_names=[],
            recorder=OpenPoseRecorder(json_directory_path=output_json_path),
        )


        os.chdir(self.openpose_exe_path)

    def run_openpose(self, input_video_filepath, output_video_folder):
        """
        Run the OpenPose demo on a video file to generate JSON outputs
        in a unique directory for each video.
        """

        # Extract video name without extension to use as a unique folder name
        video_name = Path(input_video_filepath).stem
        unique_json_output_path = self.output_json_path / video_name
        unique_json_output_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

        video_save_path = output_video_folder / f"{video_name}_openpose.avi"

        # Update the subprocess command to use the unique output directory
        subprocess.run(
            [
                "bin\OpenPoseDemo.exe",
                "--video", str(input_video_filepath),
                "--write_json", str(unique_json_output_path),
                "--net_resolution", self.net_resolution,
                "--hand",
                "--face",
                "--number_people_max", str(self.number_people_max),
                '--write_video', 
                str(video_save_path),
                "--output_resolution",
                "-1x-1",

            ],
            shell=True,
        )

    def process_video(self, input_video_filepath, output_video_folder, save_data_bool=False, use_tqdm=True):
        # Run OpenPose on the input video
        self.run_openpose(input_video_filepath, output_video_folder)

        if self.recorder is not None:
            self.recorder.record(self.tracked_objects)


    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Process an individual frame/image."""

        raise NotImplementedError("OpenPoseTracker processes video files directly and does not process individual images.")

    def annotate_image(self, image: np.ndarray, tracked_objects: Dict[str, Any], **kwargs) -> np.ndarray:
        """Annotate an image with tracking data."""

        raise NotImplementedError("OpenPoseTracker does not annotate images directly. Annotations are handled by OpenPose's output.")


if __name__ == '__main__':
    # Example usage
    from pathlib import Path

    input_video_folder = Path(r'C:\Users\aaron\FreeMocap_Data\recording_sessions\freemocap_sample_data')
    input_video_filepath = input_video_folder/'synchronized_videos'/'sesh_2022-09-19_16_16_50_in_class_jsm_synced_Cam1.mp4'
    
    output_video_folder = input_video_folder/'openpose_annotated_videos'
    output_video_folder.mkdir(parents=True, exist_ok=True)

    output_json_path = input_video_folder/'output_data'/'raw_data'/'openpose_jsons'
    output_json_path.mkdir(parents=True, exist_ok=True)

    openpose_exe_path = r'C:\openpose'
    # output_json_path = r'C:\openpose\output_json'
    # input_video_filepath = r'C:\path\to\input\video.mp4'
    # output_video_filepath = r'C:\path\to\output\video.mp4'
    tracker = OpenPoseTracker(
        openpose_exe_path=str(openpose_exe_path),
        output_json_path=str(output_json_path),
    )
    tracker.process_video(input_video_filepath, output_video_folder)