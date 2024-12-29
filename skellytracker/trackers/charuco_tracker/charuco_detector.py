from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker import BaseDetectorConfig, BaseDetector
from skellytracker.trackers.charuco_tracker.charuco_observations import CharucoObservation, CharucoObservationFactory, \
    CharucoDetection

DEFAULT_ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)


class CharucoRefinementConfig:
    window_size = (5, 5)
    zero_zone = (-1, -1)
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


class CharucoDetectorConfig(BaseDetectorConfig):
    squares_x: int = 5
    squares_y: int = 3
    aruco_dictionary: cv2.aruco.Dictionary = DEFAULT_ARUCO_DICTIONARY
    square_length: float = 1
    marker_length: float = 0.8
    refinement_config: CharucoRefinementConfig = CharucoRefinementConfig()

    @property
    def charuco_corner_ids(self) -> List[int]:
        return list(range((self.squares_x - 1) * (self.squares_y - 1)))


@dataclass
class CharucoDetector(BaseDetector):
    config: CharucoDetectorConfig
    board: cv2.aruco.CharucoBoard
    detector: cv2.aruco.CharucoDetector
    observation_factory: CharucoObservationFactory

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
            observation_factory=CharucoObservationFactory(charuco_corner_ids=config.charuco_corner_ids,
                                                          aruco_marker_ids=list(board.getIds())),
        )

    @property
    def aruco_marker_ids(self) -> List[int]:
        return list(self.board.getIds())

    @property
    def board_object_points(self) -> List[np.ndarray]:
        return list(self.board.getObjPoints())  # type: ignore

    def detect(self, image: np.ndarray) -> CharucoObservation:

        result = self._get_detection_result(image)
        if result is None:
            return self.observation_factory.create_empty_observation()
        return self.observation_factory.create_observation(result)

    def _get_detection_result(self, image: np.ndarray) -> CharucoDetection | None:
        if len(image.shape) == 2:
            grey_image = image
        else:
            grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        (raw_aruco_square_corners,
         aruco_square_ids,
         rejected_image_points) = cv2.aruco.detectMarkers(grey_image, self.config.aruco_dictionary)

        if not raw_aruco_square_corners:
            return

        # refine detected corner locations to provide sub-pixel precision
        # https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e
        refined_aruco_corners = tuple(cv2.cornerSubPix(grey_image,
                                                       corner,
                                                       winSize=self.config.refinement_config.window_size,
                                                       zeroZone=self.config.refinement_config.zero_zone,
                                                       criteria=self.config.refinement_config.termination_criteria) for
                                      corner in raw_aruco_square_corners)

        (refinement_success,
         charuco_corners,
         charuco_ids) = cv2.aruco.interpolateCornersCharuco(tuple(refined_aruco_corners),
                                                            aruco_square_ids,
                                                            grey_image,
                                                            self.board,
                                                            )


        return CharucoDetection(charuco_corner_ids=charuco_ids,
                                charuco_corners=charuco_corners,
                                aruco_marker_ids=aruco_square_ids,
                                aruco_marker_corners=refined_aruco_corners,
                                raw_aruco_corners=raw_aruco_square_corners,
                                rejected_image_points=rejected_image_points)
