from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional

import numpy as np

from skellytracker.trackers.base_tracker.tracked_object import TrackedObject

logger = logging.getLogger(__name__)


class BaseRecorder(ABC):
    """
    An abstract base class for implementing different recording algorithms.
    """

    def __init__(self):
        self.recorded_objects = []
        self.recorded_objects_array = None

    @abstractmethod
    def record(
        self, tracked_objects: Dict[str, TrackedObject], annotated_image: Optional[np.ndarray] = None
    ) -> None:
        """
        Record the tracked objects as they are created by the tracker.

        :param tracked_object: A tracked objects dictionary.
        :param annotated_image: Image array with tracking results annotated.
        :return: None
        """
        pass

    @abstractmethod
    def process_tracked_objects(self, **kwargs) -> np.ndarray:
        """
        Process the recorded objects to be in a consistent array format across trackers.

        :return: Array of tracked objects.
        """
        pass

    def clear_recorded_objects(self):
        logger.info("Clearing recorded objects from recorder")
        self.recorded_objects = []
        self.recorded_objects_array = None

    def save(self, file_path: str) -> None:
        """
        Save the recorded objects to a file.

        :param file_path: The path to the file where the recorded objects should be saved.
        :return: None
        """
        if self.recorded_objects_array is None:
            recorded_objects_array = self.process_tracked_objects()
        else:
            recorded_objects_array = self.recorded_objects_array
        logger.info(f"Saving recorded objects to {file_path}")
        np.save(file_path, recorded_objects_array)
