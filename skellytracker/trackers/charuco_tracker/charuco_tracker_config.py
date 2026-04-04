from pydantic import Field
from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseTrackerConfig, BaseDetectorConfig
from skellytracker.trackers.charuco_tracker.charuco_annotator import CharucoAnnotatorConfig
import cv2

DEFAULT_ARUCO_DICTIONARY_NAME: str = "cv2.aruco.DICT_4X4_50"
DEFAULT_ARUCO_DICTIONARY: int = cv2.aruco.DICT_4X4_50


class CharucoDetectorConfig(BaseDetectorConfig):
    squares_x: int = 5
    squares_y: int = 3
    aruco_dictionary_name: str = DEFAULT_ARUCO_DICTIONARY_NAME
    aruco_dictionary_enum: int = DEFAULT_ARUCO_DICTIONARY
    square_length: float = Field(gt=0, default=58, description="size of the edge of a black square in user-defined units (e.g., millimeters))")
    marker_length: float = Field(gt=0, le =1.0,  default=0.8, description="Length of the Aruco marker as a proportion of the square size")

    @property
    def charuco_corner_ids(self) -> list[int]:
        return list(range((self.squares_x - 1) * (self.squares_y - 1)))

    @property
    def aruco_dictionary(self) -> cv2.aruco.Dictionary:
        return cv2.aruco.getPredefinedDictionary(self.aruco_dictionary_enum)

class CharucoTrackerConfig(BaseTrackerConfig):
    detector_config: CharucoDetectorConfig = Field(default_factory = CharucoDetectorConfig)
    annotator_config: CharucoAnnotatorConfig = Field(default_factory = CharucoAnnotatorConfig)
