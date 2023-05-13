from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, Union, Optional
import numpy as np

from skelly_tracker.trackers.base_tracker.tracking_data_model import TrackingData, TrackedObject, FrameData
from skelly_tracker.trackers.webcam_demo_viewer.webcam_demo_viewer import WebcamDemoViewer

class BaseTracker(ABC):
    """
    An abstract base class for implementing different tracking algorithms.
    """
    def __init__(self, tracked_object_names: Union[list, dict], process_image_function: Callable=None, **data: Any):
        self.tracking_data = TrackingData()
        self.annotated_image = None
        self.raw_image = None
        self.tracked_objects: Dict[str, TrackedObject] = {}
        self.process_image_function: Optional[Callable] = None

        super().__init__(**data)
        if isinstance(tracked_object_names, list):
            for name in tracked_object_names:
                self.tracked_objects[name] = TrackedObject(object_id=name)
        elif isinstance(tracked_object_names, dict):
            for name, info in tracked_object_names.items():
                self.tracked_objects[name] = TrackedObject(object_id=name, **info)

        if process_image_function:
            self.process_image_function = process_image_function

    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Process the input image and apply the tracking algorithm.

        :param image: An input image.
        :return: A dictionary containing tracking data, annotated image, raw image, and tracked objects.
        """
        if self.process_image_function:
            self.process_image_function(self, image=image, **kwargs)

            return {
                "tracking_data": self.tracking_data,
                "annotated_image": self.annotated_image,
                "raw_image": self.raw_image,
                "tracked_objects": self.tracked_objects
            }

    def demo(self, window_title: str="Tracker") -> None:
        """
        Run a demo of the tracker.

        :param window_title: The title of the demo window.
        :return: None
        """
        camera_viewer = WebcamDemoViewer(self, window_title)
        camera_viewer.run()
