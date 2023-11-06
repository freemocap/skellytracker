from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union
import cv2
import numpy as np


from skelly_tracker.trackers.base_tracker.base_recorder import BaseRecorder
from skelly_tracker.trackers.base_tracker.tracked_object import TrackedObject
from skelly_tracker.trackers.image_demo_viewer.image_demo_viewer import ImageDemoViewer
from skelly_tracker.trackers.webcam_demo_viewer.webcam_demo_viewer import (
    WebcamDemoViewer,
)


class BaseTracker(ABC):
    """
    An abstract base class for implementing different tracking algorithms.
    """

    def __init__(
        self,
        tracked_object_names: List[str] = None,
        recorder: BaseRecorder = None,
        **data: Any
    ):
        self.recorder = recorder
        self.annotated_image = None
        self.raw_image = None
        self.tracked_objects: Dict[str, TrackedObject] = {}

        for name in tracked_object_names:
            self.tracked_objects[name] = TrackedObject(object_id=name)

    @abstractmethod
    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        """
        Process the input image and apply the tracking algorithm.

        :param image: An input image.
        :return: A dictionary of tracked objects
        """
        pass

    @abstractmethod
    def annotate_image(
        self, image: np.ndarray, tracked_objects: Dict[str, TrackedObject], **kwargs
    ) -> np.ndarray:
        """
        Annotate the input image with the results of the tracking algorithm.

        :param image: An input image.
        :param tracked_objects: A dictionary of tracked objects.
        :return: Annotated image
        """
        pass

    def process_video(
        self, video_filepath: Union[str, Path], save_data_bool: bool = False
    ) -> np.ndarray:
        """
        Run the tracker on a video.

        :param video_filepath: Path to video file.
        :param save_data_bool: Whether to save the data to a file.
        :return: Array of tracked keypoint data
        """

        cap = cv2.VideoCapture(str(video_filepath))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Done processing video")
                break

            image_size = (frame.shape[1], frame.shape[0])

            self.process_image(frame)
            if self.recorder is not None:
                self.recorder.record(self.tracked_objects)

        cap.release()

        if self.recorder is not None:
            output_array = self.recorder.process_tracked_objects(image_size=image_size)
            if save_data_bool:
                self.recorder.save(
                    file_path=Path(video_filepath).with_suffix(".npy")
                )
        else:
            output_array = None

        return output_array

    def demo(self) -> None:
        """
        Run a demo of the tracker.

        :param window_title: The title of the demo window.
        :return: None
        """
        camera_viewer = WebcamDemoViewer(
            tracker=self, recorder=self.recorder, window_title=self.__class__.__name__
        )
        camera_viewer.run()

    def image_demo(self, image_path: Path) -> None:
        """
        Run tracker on single image

        :return: None
        """

        image_viewer = ImageDemoViewer(self, self.__class__.__name__)
        image_viewer.run(image_path=image_path)
