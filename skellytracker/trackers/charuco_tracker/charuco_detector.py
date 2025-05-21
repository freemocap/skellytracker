import json
from typing import List

import cv2
import numpy as np
from pydantic import ConfigDict

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetectorConfig, BaseDetector
from skellytracker.trackers.charuco_tracker.charuco_observation import CharucoObservation

DEFAULT_ARUCO_DICTIONARY_NAME: str = "cv2.aruco.DICT_4X4_250"
DEFAULT_ARUCO_DICTIONARY: int = cv2.aruco.DICT_4X4_250


class CharucoDetectorConfig(BaseDetectorConfig):
    squares_x: int = 5
    squares_y: int = 3
    aruco_dictionary_name: str = DEFAULT_ARUCO_DICTIONARY_NAME
    aruco_dictionary_enum: int = DEFAULT_ARUCO_DICTIONARY
    unscaled_square_length: float = 1
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
            squareLength=config.unscaled_square_length,
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

        return CharucoObservation.from_detection_results(
            frame_number=frame_number,
            detected_charuco_corners=detected_charuco_corners,
            detected_charuco_corner_ids=detected_charuco_ids,
            detected_aruco_marker_corners=detected_aruco_corners,
            detected_aruco_marker_ids=detected_aruco_ids,
            image_size=(int(image.shape[0]), int(image.shape[1])),
            all_charuco_ids=self.config.charuco_corner_ids,
            all_aruco_ids=self.aruco_marker_ids,
            all_charuco_corners_in_object_coordinates=self.board.getChessboardCorners(),
            all_aruco_corners_in_object_coordinates=self.board.getObjPoints()
        )

    def save_board_image(self,
                     filename: str,
                     image_size: tuple[int, int] = (11000,8500),
                        margin_size: int = 10,
                     border_bits: int = 1) -> None:
        board_image = self.board.generateImage(
            outSize=image_size,
            marginSize=margin_size,
            borderBits=border_bits
        )

        annotated_board_image = cv2.putText(
            board_image,
            "Measure Charuco Square Size as the length of one size of the black squares in millimeters",
            (200, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (0, 0, 0),
            20,
            cv2.LINE_AA
        )
        annotated_board_image = cv2.putText(
            annotated_board_image,
            "Input this value into FreeMoCap to ensure correct unit scaling",
            (200, 700),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (0, 0, 0),
            20,
            cv2.LINE_AA
        )


        annotated_board_image = cv2.putText(
            annotated_board_image,
            f"Created with command:",
            (200, annotated_board_image.shape[0] - 600),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (0, 0, 0),
            20,
            cv2.LINE_AA
        )
        annotated_board_image = cv2.putText(
            annotated_board_image,
            f"`cv2.aruco.CharucoBoard(squares_x={self.config.squares_x},  squares_y={self.config.squares_y}, squareLength={self.config.unscaled_square_length}, markerLength={self.config.marker_length}, dictionary={self.config.aruco_dictionary_name})`",
            (200, annotated_board_image.shape[0] -400),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (0, 0, 0),
            20,
            cv2.LINE_AA
        )
        annotated_board_image = cv2.putText(
            annotated_board_image,
            f"using `cv2.__version__: {cv2.__version__}`",
            (200, annotated_board_image.shape[0] - 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (0, 0, 0),
            20,
            cv2.LINE_AA
        )
        cv2.imwrite(filename, annotated_board_image)


if __name__ == "__main__":
    # Example usage
    detector_config = CharucoDetectorConfig()
    detector = CharucoDetector.create(config=detector_config)
    print(detector.aruco_marker_ids)
    print(detector.board_object_points)

    detector.save_board_image("charuco_board.png")