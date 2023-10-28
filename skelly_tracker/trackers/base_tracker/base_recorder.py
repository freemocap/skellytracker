from abc import ABC, abstractmethod
from typing import Dict

from skelly_tracker.trackers.base_tracker.base_tracker import TrackedObject


class BaseRecorder(ABC):
    """
    An abstract base class for implementing different recording algorithms.
    """

    def __init__(self):
        self.recorded_objects = []

    @abstractmethod
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        """
        Record the tracked object.

        :param tracked_object: A tracked object.
        :return: None
        """
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        """
        Save the recorded objects to a file.

        :param file_path: The path to the file where the recorded objects should be saved.
        :return: None
        """
        pass