from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Callable, Optional, List

import cv2
import numpy as np

from typing import List
from dataclasses import dataclass, field

from skellytracker.trackers.image_demo_viewer.image_demo_viewer import ImageDemoViewer
from skellytracker.trackers.webcam_demo_viewer.webcam_demo_viewer import WebcamDemoViewer


@dataclass
class TrackedObject:
    """
    A dataclass for storing information about a tracked object in a single image/frame
    """
    object_id: str
    pixel_x: Optional[float] = None
    pixel_y: Optional[float] = None
    depth_z: Optional[float] = None
    extra: Optional[Dict[str, Any]] = field(default_factory=dict)


class BaseTracker(ABC):
    """
    An abstract base class for implementing different tracking algorithms.
    """

    def __init__(self,
                 tracked_object_names: List[str] = None,
                 **data: Any):
        self.annotated_image = None
        self.raw_image = None
        self.tracked_objects: Dict[str, TrackedObject] = {}
        self.results = []

        for name in tracked_object_names:
            self.tracked_objects[name] = TrackedObject(object_id=name)

    def process_video(self, video_path: str, show:bool = True, **kwargs) -> None:
        """
        Process a video and apply the tracking algorithm.

        :param video_path: A path to a video file.
        :return: None
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file {video_path} does not exist.")

        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise Exception(f"Could not open video file {video_path}.")

        success = True
        frame_idx = 0

        while success:
            success, image = video_capture.read()
            frame_idx += 1

            if not success:
                break

            self.results.append( self.process_image(image, **kwargs))

            if show:
                cv2.imshow(f"{self.__class__.__name__}", self.annotated_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    @abstractmethod
    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        """
        Process the input image and apply the tracking algorithm.

        :param image: An input image.
        :return: A dictionary of tracked objects
        """
        pass

    @abstractmethod
    def annotate_image(self, image: np.ndarray, tracked_objects: Dict[str, TrackedObject], **kwargs) -> np.ndarray:
        """
        Annotate the input image with the results of the tracking algorithm.

        :param image: An input image.
        :param tracked_objects: A dictionary of tracked objects.
        :return: Annotated image
        """
        pass

    def demo(self) -> None:
        """
        Run a demo of the tracker.

        :param window_title: The title of the demo window.
        :return: None
        """
        camera_viewer = WebcamDemoViewer(self, self.__class__.__name__)
        camera_viewer.run()

    def image_demo(self, image_path: Path) -> None:
        """
        Run tracker on single image

        :return: None
        """

        image_viewer = ImageDemoViewer(self, self.__class__.__name__)
        image_viewer.run(image_path=image_path)