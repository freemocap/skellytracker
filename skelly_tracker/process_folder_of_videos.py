from pathlib import Path

import numpy as np
from skelly_tracker.trackers.bright_point_tracker.brightest_point_tracker import BrightestPointTracker
from skelly_tracker.trackers.mediapipe_tracker.mediapipe_holistic_tracker import MediapipeHolisticTracker
from skelly_tracker.trackers.yolo_tracker.yolo_tracker import YOLOPoseTracker

file_name_dictionary = {
    MediapipeHolisticTracker: "mediapipe2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy",
    YOLOPoseTracker: "yolo2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy",
    BrightestPointTracker: "brightestPoint2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy",
}

def process_folder_of_videos(synchronized_video_path: Path, tracker) -> None:
    """
    Process a folder of synchronized videos with the given tracker. 
    Tracked data will be saved to a .npy file with the shape (numCams, numFrames, numTrackedPoints, pixelXYZ).
    
    :param synchronized_video_path: Path to folder of synchronized videos.
    :param tracker: Tracker to use.
    :return: None
    """
    file_name = file_name_dictionary[tracker]
    output_path = synchronized_video_path.parent / "output_data" / "raw_data" / file_name
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    array_list = []
    for video_path in synchronized_video_path.glob("*.mp4"):
        output_array = tracker.process_video(video_filepath=video_path, save_data_bool=False)
        array_list.append(output_array)

    combined_array = np.stack(array_list)

    print(f"Shape of output array: {combined_array.shape}")
    np.save(output_path, combined_array)

if __name__ == "__main__":
    synchronized_video_path = Path("skelly_tracker/data/synchronized_videos")
    tracker = YOLOPoseTracker()
    process_folder_of_videos(synchronized_video_path, tracker)