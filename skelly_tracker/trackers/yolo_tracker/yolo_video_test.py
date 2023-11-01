from pathlib import Path

import numpy as np
from skelly_tracker.trackers.yolo_tracker.yolo_tracker import YOLOPoseTracker


if __name__ == "__main__":
    synchronized_video_path = Path("")
    output_path = synchronized_video_path.parent / "output_data" / "raw_data" / "mediapipe2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
    
    array_list = []
    for video_path in synchronized_video_path.glob("*.mp4"):
        output_array = YOLOPoseTracker().process_video(video_filepath=video_path, save_data_bool=False)
        array_list.append(output_array)

    combined_array = np.stack(array_list)

    print(combined_array.shape)
    np.save(output_path, combined_array)