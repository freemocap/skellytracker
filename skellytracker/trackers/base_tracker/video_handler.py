import logging
from pathlib import Path
from typing import Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoHandler:
    def __init__(
        self,
        output_path: Union[Path, str],
        frame_size: tuple[int, int],
        fps: float = 30.0,
        codec: str = "mp4v",
    ):
        """
        Initialize the VideoHandler.

        :param output_path: The path to the output video file.
        :param frame_size: The size of the frames (width, height).
        :param fps: The frames per second of the output video.
        :param codec: The codec to use for the output video.
        """
        self.output_path = output_path
        fourcc = cv2.VideoWriter.fourcc(*codec)
        self.video_writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, frame_size
        )

    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add a frame to the video.

        :param frame: The frame to add.
        """
        self.video_writer.write(frame)

    def close(self):
        """
        Close the video file.
        """
        self.video_writer.release()
        logger.info(f"video saved to {self.output_path}")
