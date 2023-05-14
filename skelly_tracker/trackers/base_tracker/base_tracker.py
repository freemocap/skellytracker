from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, Optional, List

import numpy as np

from skelly_tracker.trackers.webcam_demo_viewer.webcam_demo_viewer import WebcamDemoViewer


class BaseTracker(ABC):
    """
    An abstract base class for implementing different tracking algorithms.
    """

    def __init__(self, tracked_object_names: List[str], **data: Any):
        self.tracking_data = {}
        self.annotated_image = None
        self.raw_image = None
        self.tracked_objects: Dict[str, Dict] = {}
        self.process_image_function: Optional[Callable] = None

        super().__init__()

        for name in tracked_object_names:
            self.tracked_objects[name] = {"object_id": name, "pixel": {"x": None, "y": None}}


    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Process the input image and apply the tracking algorithm.

        :param image: An input image.
        :return: A dictionary of the form {"tracking_data": tracking_data, "annotated_image": annotated_image, "raw_image": raw_image}
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
