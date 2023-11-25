import os
import subprocess
from pathlib import Path
from typing import Union

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker


class OpenposeTracker:
    net_resolution: str = "-1x368"
    track_hands = True
    track_face = True
    number_people_max = 1
    openpose_base_directory: Union[str, Path] = Path().home() / "openpose"
    open_pose_exe_path: str = "bin/OpenPoseDemo.exe"

    def __init__(self,
                 video_path: str,
                 save_path: str= None):
        if not os.name == "nt":
            raise OSError("OpenPoseTracker only runs on Windows (for now)")

        super().__init__()
        self._video_path = Path(video_path)

        if save_path is None:
            self._save_path = Path(video_path).parent
        else:
            self._save_path = Path(save_path)

    def process_video(self, **kwargs) -> None:
        json_path = Path(self._save_path) / "json"
        json_path.mkdir(parents=True, exist_ok=True)
        open_pose_exe_path_full = self.openpose_base_directory / self.open_pose_exe_path
        if not open_pose_exe_path_full.exists():
            raise FileNotFoundError(f"Could not find OpenPose executable at {open_pose_exe_path_full}")
        subprocess.run(
            [
                str(open_pose_exe_path_full),
                "--video",
                str(self._video_path),
                "--write_json",
                str(json_path),
                "--net_resolution",
                self.net_resolution,
                "--hand" if self.track_hands else "",
                "--face" if self.track_face else "",
                "--number_people_max",
                str(self.number_people_max),
                "-write_video",
                str(Path(self._save_path) / (str(self._video_path.stem) + "_openPose.avi")),
            ],
            cwd=str(self.openpose_base_directory),
            shell=True,
        )


if __name__ == "__main__":
    tacker = OpenposeTracker(video_path=r"C:\Users\jonma\freemocap_data\recording_sessions\session_2023-11-15_15_14_59\recording_15_18_22_gmt-5\synchronized_videos\Camera_000_synchronized.mp4")
    tacker.process_video()
    print("done!")
