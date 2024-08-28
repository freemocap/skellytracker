import subprocess
from pathlib import Path
from typing import Optional, Union
from skellytracker.trackers.base_tracker.base_tracker import BaseCumulativeTracker
from skellytracker.trackers.openpose_tracker.openpose_recorder import OpenPoseRecorder


class OpenPoseTracker(BaseCumulativeTracker):
    def __init__(
        self,
        openpose_root_folder_path: Union[str, Path],
        output_json_folder_path: Optional[Union[str, Path]] = None,
        net_resolution: str = "-1x320",
        number_people_max: int = 1,
        track_hands: bool = True,
        track_faces: bool = True,
        output_resolution: str = "-1x-1",
    ):
        """
        Initialize the OpenPoseTracker.

        :param recorder: An instance of OpenPoseRecorder for handling the output.
        :param openpose_root_folder_path: Path to the OpenPose root folder.
        :param output_json_folder_path: Path to the output JSON folder.
        :param net_resolution: Network resolution for OpenPose processing.
        :param number_people_max: Maximum number of people to detect.
        :param track_hands: Whether to track hands.
        :param track_faces: Whether to track faces.
        :param output_resolution: Output resolution for video.
        """
        super().__init__(
            tracked_object_names=[],
            recorder=OpenPoseRecorder(
                track_hands=track_hands,
                track_faces=track_faces,
            ),
            track_hands=track_hands,
            track_faces=track_faces,
        )
        self.openpose_root_folder_path = Path(openpose_root_folder_path)
        self.output_json_folder_path = output_json_folder_path
        self.net_resolution = net_resolution
        self.number_people_max = number_people_max
        self.track_hands = track_hands
        self.track_faces = track_faces
        self.output_resolution = output_resolution

    def set_track_hands(self, track_hands: bool):
        self._track_hands = track_hands
        self.recorder.track_hands = track_hands

    def set_track_faces(self, track_faces: bool):
        self._track_faces = track_faces
        self.recorder.track_faces = track_faces

    def set_json_output_path(self, output_json_folder_path: Union[str, Path]):
        self.output_json_folder_path = Path(output_json_folder_path)

    def process_video(
        self,
        input_video_filepath: Union[str, Path],
        output_video_filepath: Union[str, Path],
        save_data_bool: bool = False,
        use_tqdm: bool = True,  # TODO: this is unused, replace with an openpose flag or remove
        **kwargs,
    ):
        """
        Run the OpenPose demo on a video file to generate JSON outputs
        in a unique directory for each video.

        :param input_video_filepath: Path to the input video file.
        :param output_video_filepath: Path to the output video file.
        :param save_data_bool: Whether to save the data.
        :param use_tqdm: Whether to use tqdm progress bar.
        :return: The output array, or None if recorder isn't initialized in tracker.
        """
        # Extract video name without extension to use as a unique folder name
        video_name = Path(input_video_filepath).stem

        if self.output_json_folder_path is None:
            self.output_json_folder_path = Path(input_video_filepath).parent.parent / "output_data" / "raw_data" / "openpose_jsons"

        Path(self.output_json_folder_path).mkdir(parents=True, exist_ok=True)

        unique_json_output_path = Path(self.output_json_folder_path) / video_name
        unique_json_output_path.mkdir(parents=True, exist_ok=True)

        # Full path to the OpenPose executable
        openpose_executable_path = (
            self.openpose_root_folder_path / "bin" / "OpenPoseDemo.exe"
        )

        openpose_command = [
            str(openpose_executable_path),  # Full path to the OpenPose executable
            "--video",
            str(input_video_filepath),
            "--write_json",
            str(unique_json_output_path),
            "--net_resolution",
            str(self.net_resolution),
            "--number_people_max",
            str(self.number_people_max),
            "--write_video",
            str(output_video_filepath),
            "--output_resolution",
            str(self.output_resolution),
        ]

        if self.track_hands:
            openpose_command.append("--hand")
        if self.track_faces:
            openpose_command.append("--face")

        # Update the subprocess command to use the unique output directory
        try:
            subprocess.run(  # noqa: S603
                openpose_command,
                shell=False,
                cwd=self.openpose_root_folder_path,  # Set the current working directory for the subprocess
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            return None

        if self.recorder is not None:
            output_array = self.recorder.process_tracked_objects(
                output_json_path=unique_json_output_path
            )
            if save_data_bool:
                self.recorder.save(
                    file_path=str(Path(input_video_filepath).with_suffix(".npy"))
                )
        else:
            output_array = None

        return output_array


if __name__ == "__main__":
    # Example usage
    openpose_root_folder_path = r"C:\openpose"
    input_video_filepath = r'C:\path\to\input\video.mp4'
    output_video_filepath = r'C:\path\to\output\video.avi'

    tracker = OpenPoseTracker(
        openpose_root_folder_path=str(openpose_root_folder_path),
    )
    tracker.process_video(input_video_filepath, output_video_filepath)
