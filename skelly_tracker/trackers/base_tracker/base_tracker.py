from abc import ABC, abstractmethod
from dataclasses import Field
from typing import Any, Dict, Callable, Union
import numpy as np
from pydantic import BaseModel
from skelly_tracker.trackers.base_tracker.tracking_data_model import TrackingData, TrackedObject, FrameData

from skelly_tracker.trackers.webcam_demo_viewer.webcam_demo_viewer import CameraViewer

class BaseTracker(ABC, BaseModel):
    """
    An abstract base class for implementing different tracking algorithms.
    """
    tracking_data: TrackingData = TrackingData()
    annotated_image: np.ndarray = None
    raw_image: np.ndarray = None
    tracked_objects: Dict[str, TrackedObject] = {}
    process_image_function: Callable = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, tracked_object_names: Union[list, dict], process_image_func: Callable, **data: Any):
        super().__init__(**data)
        if isinstance(tracked_object_names, list):
            for name in tracked_object_names:
                self.tracked_objects[name] = TrackedObject(object_id=name)
        elif isinstance(tracked_object_names, dict):
            for name, info in tracked_object_names.items():
                self.tracked_objects[name] = TrackedObject(object_id=name, **info)
        self.process_image_function = process_image_func

    def process_image(self, image, **kwargs):
        """
        Process the input image and apply the tracking algorithm.

        :param image: An input image.
        :return: None
        """
        self.process_image_function(self, image=image, **kwargs)

        return {
            "tracking_data": self.tracking_data,
            "annotated_image": self.annotated_image,
            "raw_image": self.raw_image,
            "tracked_objects": self.tracked_objects
        }

    def demo(self, window_title="Tracker"):
        camera_viewer = CameraViewer(self, window_title)
        camera_viewer.run()

