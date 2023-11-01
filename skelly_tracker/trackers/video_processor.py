from pathlib import Path
from typing import Union
import cv2
import numpy as np


class VideoProcessor:
    def __init__(self, tracker, recorder = None, video_filepath: Union[str, Path] = None):
        self.tracker = tracker
        self.recorder = recorder
        self.video_filepath = video_filepath
        self.data_array = None

    def run(self, save_data_bool = False) -> np.ndarray:
        """
        Run the tracker on a video.
        
        :save_data_bool: Whether to save the data to a file.
        :return: None
        """
        cap = cv2.VideoCapture(str(self.video_filepath))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            self.tracker.process_image(frame)
            if self.recorder is not None:
                self.recorder.record(self.tracker.tracked_objects)

        cap.release()
        
        if self.recorder is not None:
            self.data_array = self.recorder.process_tracked_objects()
            if save_data_bool:
                self.recorder.save(file_path = Path(self.video_filepath).with_suffix(".npy"))

        return self.data_array
            