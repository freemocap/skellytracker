from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from skelly_tracker.trackers.base_tracker.base_recorder import BaseRecorder
from skelly_tracker.trackers.base_tracker.tracked_object import TrackedObject
from skelly_tracker.trackers.image_demo_viewer.image_demo_viewer import ImageDemoViewer
from skelly_tracker.trackers.webcam_demo_viewer.webcam_demo_viewer import WebcamDemoViewer
from typing import List


class BaseTracker(ABC):
    """
    An abstract base class for implementing different tracking algorithms.
    """

    def __init__(self,
                 tracked_object_names: List[str]=None,
                 recorder: BaseRecorder=None,
                 **data: Any):
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
        camera_viewer = WebcamDemoViewer(tracker=self,
                                         recorder=self.recorder,
                                         window_title=self.__class__.__name__)
        camera_viewer.run()

    def image_demo(self, image_path: Path) -> None:
        """
        Run tracker on single image
        
        :return: None
        """

        image_viewer = ImageDemoViewer(self, self.__class__.__name__)
        image_viewer.run(image_path=image_path)