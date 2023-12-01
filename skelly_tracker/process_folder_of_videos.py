import logging
from pathlib import Path
from typing import Optional
import numpy as np


from skelly_tracker.trackers.base_tracker.base_tracker import BaseTracker
from skelly_tracker.trackers.bright_point_tracker.brightest_point_tracker import (
    BrightestPointTracker,
)
from skelly_tracker.trackers.mediapipe_tracker.mediapipe_holistic_tracker import (
    MediapipeHolisticTracker,
)
from skelly_tracker.trackers.yolo_mediapipe_combo_tracker.yolo_mediapipe_combo_tracker import (
    YOLOMediapipeComboTracker,
)
from skelly_tracker.trackers.yolo_tracker.yolo_tracker import YOLOPoseTracker

logger = logging.getLogger(__name__)

file_name_dictionary = {
    "MediapipeHolisticTracker": "mediapipe2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy",
    "YOLOMediapipeComboTracker": "mediapipe2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy",
    "YOLOPoseTracker": "yolo2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy",
    "BrightestPointTracker": "brightestPoint2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy",
}


def process_folder_of_videos(
    tracker: BaseTracker,
    synchronized_video_path: Path,
    output_path: Optional[Path] = None,
    annotated_video_path: Optional[Path] = None,
) -> None:
    """
    Process a folder of synchronized videos with the given tracker.
    Tracked data will be saved to a .npy file with the shape (numCams, numFrames, numTrackedPoints, pixelXYZ).

    :param synchronized_video_path: Path to folder of synchronized videos.
    :param tracker: Tracker to use.
    :return: Array of tracking data
    """
    file_name = file_name_dictionary[tracker.__class__.__name__]
    if output_path is None:
        output_path = (
            synchronized_video_path.parent / "output_data" / "raw_data" / file_name
        )
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if annotated_video_path is None:
        annotated_video_path = synchronized_video_path.parent / "annotated_videos"
    if not annotated_video_path.exists():
        annotated_video_path.mkdir(parents=True, exist_ok=True)

    array_list = []
    for video_path in synchronized_video_path.glob("*.mp4"):
        video_name = video_path.stem + "_mediapipe.mp4" # TODO: fix it so blender output doesn't require mediapipe addendum here
        output_array = tracker.process_video(
            input_video_filepath=video_path,
            output_video_filepath=annotated_video_path / video_name,
            save_data_bool=False,
        )
        array_list.append(output_array)
        tracker.recorder.clear_recorded_objects()

    combined_array = np.stack(array_list)

    logger.info(f"Shape of output array: {combined_array.shape}")
    np.save(output_path, combined_array)

    return combined_array


if __name__ == "__main__":
    synchronized_video_path = Path(
        "/Users/philipqueen/freemocap_data/recording_sessions/sesh_2023-05-10_16_47_16_parade_AC_MN_JSM_ML/synchronized_videos"
    )
    tracker = YOLOMediapipeComboTracker()
    process_folder_of_videos(tracker=tracker, synchronized_video_path=synchronized_video_path)
