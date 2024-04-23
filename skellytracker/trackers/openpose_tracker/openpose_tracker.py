import subprocess
from pathlib import Path
from typing import Union
from skellytracker.trackers.base_tracker.base_tracker import BaseCumulativeTracker
from skellytracker.trackers.openpose_tracker.openpose_recorder import OpenPoseRecorder

import os


class OpenPoseTracker(BaseCumulativeTracker):
    def __init__(
        self,
        openpose_exe_path: Union[str, Path],
        output_json_path: Union[str, Path],
        net_resolution: str = "-1x640",
        number_people_max: int = 1,
    ):
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
        self.net_resolution = net_resolution # TODO: this and num_people should be parameters for process_video, since we could use this one tracker to process videos with different parameters
        self.number_people_max = number_people_max

        super().__init__(
            tracked_object_names=[],
            recorder=OpenPoseRecorder(json_directory_path=output_json_path),
        )

        os.chdir(self.openpose_exe_path) # TODO: this can mess with things downstream if we're not sure to change the path back, we should just be explicit about the path throughout

    def process_video(
        self,
        input_video_filepath: Union[str, Path],
        output_video_filepath: Union[str, Path],
        save_data_bool: bool = False,
        use_tqdm: bool = True,
        **kwargs,
    ):
        """
        Run the OpenPose demo on a video file to generate JSON outputs
        in a unique directory for each video.
        """

        # TODO: Add some check that the openpose executable exists (and maybe do a cross platform search like we do for blender in fmc?)

        # Extract video name without extension to use as a unique folder name
        video_name = Path(input_video_filepath).stem
        unique_json_output_path = self.output_json_path / video_name
        unique_json_output_path.mkdir(
            parents=True, exist_ok=True
        )  # Create the directory if it doesn't exist

        # video_save_path = Path(output_video_filepath) / f"{video_name}_openpose.avi"

        # Update the subprocess command to use the unique output directory
        # TODO: subprocess call should probably be in a try/except with some form of error handling
        subprocess.run(
            [
                "bin\OpenPoseDemo.exe",  # TODO: this is throwing the error `S607 Starting a process with a partial executable path`, needs to be something like usr/bin/OpenPoseDemo.exe, but I can't verify where it is (also does it need two backslashes to escape properly?)
                "--video",
                str(input_video_filepath),
                "--write_json",
                str(unique_json_output_path),
                "--net_resolution",
                self.net_resolution,
                "--hand",
                "--face",
                "--number_people_max",
                str(self.number_people_max),
                "--write_video",
                str(output_video_filepath),
                "--output_resolution",
                "-1x-1",
            ],
            shell=True,  # TODO: This is generally frowned upon if there's not a reason for it, see `S602 subprocess call with shell=True identified, security issue.`
        )


if __name__ == "__main__":
    # Example usage

    input_video_folder = Path(
        r"C:\Users\aaron\FreeMocap_Data\recording_sessions\freemocap_sample_data"
    )
    input_video_filepath = (
        input_video_folder
        / "synchronized_videos"
        / "sesh_2022-09-19_16_16_50_in_class_jsm_synced_Cam1.mp4"
    )

    output_video_filepath = input_video_folder / "openpose_annotated_videos"
    output_video_filepath.mkdir(parents=True, exist_ok=True)

    output_json_path = (
        input_video_folder / "output_data" / "raw_data" / "openpose_jsons"
    )
    output_json_path.mkdir(parents=True, exist_ok=True)

    openpose_exe_path = r"C:\openpose"
    # output_json_path = r'C:\openpose\output_json'
    # input_video_filepath = r'C:\path\to\input\video.mp4'
    # output_video_filepath = r'C:\path\to\output\video.mp4'
    tracker = OpenPoseTracker(
        openpose_exe_path=str(openpose_exe_path),
        output_json_path=str(output_json_path),
    )
    tracker.process_video(input_video_filepath, output_video_filepath)
