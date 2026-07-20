from typing import Literal

import cv2
from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseTrackerConfig, BaseDetectorConfig, TrackerType
from skellytracker.trackers.charuco_tracker.charuco_annotator import CharucoAnnotatorConfig
from skellytracker.trackers.charuco_tracker.charuco_board_definition import CharucoBoardDefinition


class CharucoDetectorConfig(BaseDetectorConfig):
    tracker_type: Literal[TrackerType.CHARUCO] = TrackerType.CHARUCO
    board: CharucoBoardDefinition = Field(default_factory=CharucoBoardDefinition.create_letter_size_5x3)


    @property
    def squares_x(self) -> int:
        return self.board.squares_x

    @property
    def squares_y(self) -> int:
        return self.board.squares_y

    @property
    def square_length(self) -> float:
        return self.board.square_length_mm

    @property
    def marker_length(self) -> float:
        return self.board.marker_length_ratio

    @property
    def aruco_dictionary_enum(self) -> int:
        return self.board.aruco_dictionary_enum

    @property
    def aruco_dictionary(self) -> cv2.aruco.Dictionary:
        return self.board.aruco_dictionary

    @property
    def aruco_dictionary_name(self) -> str:
        return f"cv2.aruco dict enum {self.board.aruco_dictionary_enum}"

    @property
    def charuco_corner_ids(self) -> list[int]:
        return list(range(self.board.n_corners))


class CharucoTrackerConfig(BaseTrackerConfig):
    detector_config: CharucoDetectorConfig = Field(default_factory=CharucoDetectorConfig)
    annotator_config: CharucoAnnotatorConfig = Field(default_factory=CharucoAnnotatorConfig)
