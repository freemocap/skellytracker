from typing import List

import cv2
import numpy as np
from pydantic import ConfigDict

from skellytracker.trackers.base_tracker.base_tracker import BaseDetectorConfig, BaseDetector
from skellytracker.trackers.charuco_tracker.charuco_observation import CharucoObservation

DEFAULT_ARUCO_DICTIONARY:int = cv2.aruco.DICT_4X4_250


class CharucoDetectorConfig(BaseDetectorConfig):
    squares_x: int = 5
    squares_y: int = 3
    aruco_dictionary_enum:int  = DEFAULT_ARUCO_DICTIONARY
    square_length: float = 1
    marker_length: float = 0.8

    @property
    def charuco_corner_ids(self) -> List[int]:
        return list(range((self.squares_x - 1) * (self.squares_y - 1)))

    @property
    def aruco_dictionary(self) -> cv2.aruco.Dictionary:
        return cv2.aruco.getPredefinedDictionary(self.aruco_dictionary_enum)

class CharucoDetector(BaseDetector):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: CharucoDetectorConfig
    board: cv2.aruco.CharucoBoard
    detector: cv2.aruco.CharucoDetector

    @classmethod
    def create(cls, config: CharucoDetectorConfig):
        board = cv2.aruco.CharucoBoard(
            size=(config.squares_x, config.squares_y),
            squareLength=config.square_length,
            markerLength=config.marker_length,
            dictionary=config.aruco_dictionary,
        )
        detector = cv2.aruco.CharucoDetector(board)
        return cls(
            config=config,
            board=board,
            detector=detector,
        )

    @property
    def aruco_marker_ids(self) -> List[int]:
        return list(self.board.getIds())

    @property
    def board_object_points(self) -> List[np.ndarray]:
        return list(self.board.getObjPoints())  # type: ignore

    def detect(self,
                frame_number: int,
               image: np.ndarray) -> CharucoObservation:
        if len(image.shape) == 2:
            grey_image = image
        else:
            grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (detected_charuco_corners,
         detected_charuco_ids,
         detected_aruco_corners,
         detected_aruco_ids) = self.detector.detectBoard(grey_image)

        return CharucoObservation.from_detect_board_results(detected_charuco_corners=detected_charuco_corners,
                                                            detected_charuco_corner_ids=detected_charuco_ids,
                                                            detected_aruco_marker_corners=detected_aruco_corners,
                                                            detected_aruco_marker_ids=detected_aruco_ids,
                                                            image_size=(int(image.shape[0]), int(image.shape[1])),
                                                            all_charuco_ids=self.config.charuco_corner_ids,
                                                            all_aruco_ids=self.aruco_marker_ids,
                                                            all_charuco_corners_in_object_coordinates=self.board.getChessboardCorners(),
                                                            all_aruco_corners_in_object_coordinates=self.board.getObjPoints()
                                                            )
