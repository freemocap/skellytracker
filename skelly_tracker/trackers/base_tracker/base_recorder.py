from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from skelly_tracker.trackers.base_tracker.tracked_object import TrackedObject


class BaseRecorder(ABC):
    """
    An abstract base class for implementing different recording algorithms.
    """

    def __init__(self):
        self.recorded_objects = []
        self.recorded_objects_array = None

    @abstractmethod
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        """
        Record the tracked objects as they are created by the tracker.

        :param tracked_object: A tracked objects dictionary.
        :return: None
        """
        pass

    @abstractmethod
    def process_tracked_objects(self) -> np.ndarray:
        """
        Process the recorded objects to be in a consistent array format across trackers.

        :return: Array of tracked objects.
        """
        pass

    def save(self, file_path: str) -> None:
        """
        Save the recorded objects to a file.

        :param file_path: The path to the file where the recorded objects should be saved.
        :return: None
        """
        if self.recorded_objects_array is None:
            self.process_tracked_objects()
        print(f"Saving recorded objects to {file_path}")
        np.save(file_path, self.recorded_objects_array)