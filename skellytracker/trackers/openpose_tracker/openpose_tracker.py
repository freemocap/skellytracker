import subprocess
from pathlib import Path
from typing import Union
from skellytracker.trackers.base_tracker.base_tracker import BaseCumulativeTracker
from skellytracker.trackers.openpose_tracker.openpose_recorder import OpenPoseRecorder
import os

class OpenPoseTracker(BaseCumulativeTracker):
    def __init__(
        self,
        openpose_root_folder_path: Union[str, Path],
        output_json_path: Union[str, Path],
    ):
        """
        Initialize the OpenPoseTracker.

        :param recorder: An instance of OpenPoseRecorder for handling the output.
        :param openpose_root_folder_path: Path to the OpenPose root folder.
        :param output_json_path: Directory where JSON files will be saved.
        """
        super().__init__(
            tracked_object_names=[],
            recorder=OpenPoseRecorder(json_directory_path=output_json_path),
        )
        self.openpose_root_folder_path = Path(openpose_root_folder_path)
        self.output_json_path = Path(output_json_path)

    def process_video(
        self,
        input_video_filepath: Union[str, Path],
        output_video_filepath: Union[str, Path],
        net_resolution: str = "-1x640",
        number_people_max: int = 1,
        track_hands: bool = True,
        track_faces: bool = True,
        save_data_bool: bool = False,
        use_tqdm: bool = True,  # TODO: this is unused, replace with an openpose flag or remove
        **kwargs,
    ):
        """
        Run the OpenPose demo on a video file to generate JSON outputs
        in a unique directory for each video.

        :param input_video_filepath: Path to the input video file.
        :param output_video_filepath: Path to the output video file.
        :param net_resolution: Network resolution for OpenPose processing.
        :param number_people_max: Maximum number of people to detect.
        :param track_hands: Whether to track hands.
        :param track_faces: Whether to track faces.
        :param save_data_bool: Whether to save the data.
        :param use_tqdm: Whether to use tqdm progress bar.
        :return: The output array, or None if recorder isn't initialized in tracker.
        """
        # Extract video name without extension to use as a unique folder name
        video_name = Path(input_video_filepath).stem
        unique_json_output_path = self.output_json_path / video_name
        unique_json_output_path.mkdir(parents=True, exist_ok=True)

        # Full path to the OpenPose executable
        openpose_executable_path = self.openpose_root_folder_path / "bin" / "OpenPoseDemo.exe"

        openpose_command = [
                    str(openpose_executable_path),  # Full path to the OpenPose executable
                    "--video",
                    str(input_video_filepath),
                    "--write_json",
                    str(unique_json_output_path),
                    "--net_resolution",
                    net_resolution,
                    "--number_people_max",
                    str(number_people_max),
                    "--write_video",
                    str(output_video_filepath),
                    "--output_resolution",
                    "-1x-1",
                ]

        if track_hands:
            openpose_command.append("--hand")
        if track_faces:
            openpose_command.append("--face")

        # Update the subprocess command to use the unique output directory
        try:
            result = subprocess.run(
                openpose_command,
                shell=False,
                cwd=self.openpose_root_folder_path  # Set the current working directory for the subprocess
            )

            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, result.args)

        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            return None

        if self.recorder is not None:
            output_array = self.recorder.process_tracked_objects(output_json_path=unique_json_output_path)
            if save_data_bool:
                self.recorder.save(
                    file_path=str(Path(input_video_filepath).with_suffix(".npy"))
                )
        else:
            output_array = None

        return output_array
            



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

    openpose_root_folder_path = r"C:\openpose"
    # output_json_path = r'C:\openpose\output_json'
    # input_video_filepath = r'C:\path\to\input\video.mp4'
    # output_video_filepath = r'C:\path\to\output\video.mp4'
    tracker = OpenPoseTracker(
        openpose_root_folder_path=str(openpose_root_folder_path),
        output_json_path=str(output_json_path),
    )
    tracker.process_video(input_video_filepath, output_video_filepath)
