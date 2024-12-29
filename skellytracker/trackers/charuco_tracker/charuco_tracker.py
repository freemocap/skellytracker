from dataclasses import field, dataclass
from typing import List

import cv2
import numpy as np
from pydantic import BaseModel

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker, BaseTrackerConfig, BaseObservation, \
    BaseDetectorConfig, BaseImageAnnotatorConfig, BaseDetector

DEFAULT_ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)


@dataclass
class CharucoObservation(BaseObservation):
    charuco_corners: dict[int, tuple[float, float] | None]
    aruco_marker_corners: dict[int, list[tuple[float, float]] | None]


CharucoObservations = list[CharucoObservation]


class CharucoDetectorConfig(BaseDetectorConfig):
    squares_x: int = 5
    squares_y: int = 3
    aruco_dictionary: cv2.aruco.Dictionary = DEFAULT_ARUCO_DICTIONARY
    square_length: float = 1
    marker_length: float = 0.8

    @property
    def charuco_corner_ids(self) -> List[int]:
        return list(range((self.squares_x - 1) * (self.squares_y - 1)))

    @property
    def charuco_corner_names(self) -> List[str]:
        return [f"CharucoCorner_{index}" for index in self.charuco_corner_ids]


@dataclass
class CharucoObservationFactory:
    charuco_corner_ids: List[int]
    aruco_marker_ids: List[int]

    def create_observation(self,
                           detected_charuco_corners: np.ndarray,
                           detected_charuco_corner_ids: list[list[int]],
                           detected_aruco_marker_corners: np.ndarray,
                           detected_aruco_marker_ids: list[list[int]],
                           ) -> CharucoObservation:
        charuco_corners_out = {id: None for id in self.charuco_corner_ids}
        aruco_marker_corners_out = {id: None for id in self.aruco_marker_ids}

        if detected_charuco_corner_ids is None or len(detected_charuco_corner_ids) == 0:
            detected_charuco_corner_ids = None
            detected_charuco_corners = None
        elif len(detected_charuco_corner_ids) == 1:
            detected_charuco_corner_ids = list(detected_charuco_corner_ids[0])
            detected_charuco_corners = list(detected_charuco_corners[0])
        elif len(detected_charuco_corner_ids) > 1:
            detected_charuco_corner_ids = list(np.squeeze(detected_charuco_corner_ids))
            detected_charuco_corners = list(np.squeeze(detected_charuco_corners))

        if detected_aruco_marker_ids is None or len(detected_aruco_marker_ids) == 0:
            detected_aruco_marker_ids = None
            detected_aruco_marker_corners = None
        elif len(detected_aruco_marker_ids) == 1:
            detected_aruco_marker_ids = list(detected_aruco_marker_ids[0])
            detected_aruco_marker_corners = list(detected_aruco_marker_corners[0])
        elif len(detected_aruco_marker_ids) > 1:
            detected_aruco_marker_ids = list(np.squeeze(detected_aruco_marker_ids))
            detected_aruco_marker_corners = list(np.squeeze(detected_aruco_marker_corners))

        if detected_charuco_corner_ids is not None:
            for corner_id, corner in zip(detected_charuco_corner_ids, detected_charuco_corners):
                charuco_corners_out[corner_id] = corner

        if detected_aruco_marker_ids is not None:
            for marker_id, corner in zip(detected_aruco_marker_ids, detected_aruco_marker_corners):
                aruco_marker_corners_out[marker_id] = corner

        return CharucoObservation(
            charuco_corners=charuco_corners_out,
            aruco_marker_corners=aruco_marker_corners_out,
        )


class CharucoAnnotatorConfig(BaseImageAnnotatorConfig):
    marker_type: int = cv2.MARKER_CROSS
    marker_size: int = 30
    marker_thickness: int = 2
    marker_color: tuple[int, int, int] = (0, 0, 255)

    text_color: tuple[int, int, int] = (255, 0, 0)
    text_size: float = 1
    text_thickness: int = 2
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX


class CharucoTrackerConfig(BaseTrackerConfig):
    detector_config: CharucoDetectorConfig
    annotator_config: CharucoAnnotatorConfig

    @classmethod
    def create(cls):
        return cls(detector_config=CharucoDetectorConfig(),
                   annotator_config=CharucoAnnotatorConfig())


class CharucoImageAnnotator(BaseModel):
    config: CharucoAnnotatorConfig

    @classmethod
    def create(cls, config: CharucoAnnotatorConfig):
        return cls(config=config)

    def annotate_image(
            self,
            image: np.ndarray,
            observations: CharucoObservations,
            show_tracks: int | None = 10,
    ) -> np.ndarray:
        # Copy the original image for annotation
        annotated_image = image.copy()

        if show_tracks is not None:
            show_tracks = min(show_tracks, len(observations))
        else:
            show_tracks = len(observations)

        # Draw a marker for each tracked corner
        for obs_count, observation in enumerate(observations[-show_tracks:]):
            for corner_id, corner in observation.charuco_corners.items():
                if obs_count == 0:
                    marker_color = self.config.marker_color
                    marker_thickness = self.config.marker_thickness
                    marker_size = self.config.marker_size
                else:
                    obs_count_scale = 1 - (obs_count / show_tracks)
                    marker_color = tuple(element * obs_count_scale for element in self.config.marker_color)
                    marker_thickness = max(1,int(self.config.marker_thickness * obs_count_scale))
                    marker_size = max(1, int(self.config.marker_size * obs_count_scale))



                if corner is not None:
                    cv2.drawMarker(
                        annotated_image,
                        (int(corner[0]), int(corner[1])),
                        marker_color,
                        markerType=self.config.marker_type,
                        markerSize=marker_size,
                        thickness=marker_thickness,
                    )
                    if obs_count == 0:
                        cv2.putText(
                            annotated_image,
                            f"ID: {corner_id}",
                            (int(corner[0]), int(corner[1])),
                            self.config.text_font,
                            self.config.text_size,
                            self.config.text_color,
                            self.config.text_thickness,
                        )

        # List undetected markers
        undetected_markers = [
            marker_id for marker_id, corners in observations[-1].aruco_marker_corners.items() if corners is None
        ]
        undetected_text = "Undetected markers: " + ", ".join(map(str, undetected_markers))

        # Put the undetected markers text in the upper right corner
        text_size, _ = cv2.getTextSize(undetected_text, self.config.text_font, self.config.text_size,
                                       self.config.text_thickness)
        text_x = annotated_image.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10
        cv2.putText(
            annotated_image,
            undetected_text,
            (text_x, text_y),
            self.config.text_font,
            self.config.text_size,
            self.config.text_color,
            self.config.text_thickness,
        )

        return annotated_image


@dataclass
class CharucoDetector(BaseDetector):
    config: CharucoDetectorConfig
    board: cv2.aruco.CharucoBoard
    detector: cv2.aruco.CharucoDetector  # https://docs.opencv.org/4.10.0/d9/df5/classcv_1_1aruco_1_1CharucoDetector.html
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
        if len(image.shape) == 2:
            grey_image = image
        else:
            grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = self.detector.detectBoard(grey_image)
        return self.observation_factory.create_observation(*result)


@dataclass
class CharucoTracker(BaseTracker):
    config: CharucoTrackerConfig
    detector: CharucoDetector
    annotator: CharucoImageAnnotator
    observations: CharucoObservations = field(default_factory=list)

    @classmethod
    def create(cls, config: CharucoTrackerConfig | None = None):
        if config is None:
            config = CharucoTrackerConfig.create()
        # save_aruco_dictionary(config.detector_config.aruco_dictionary)
        return cls(
            config=config,
            detector=CharucoDetector.create(config.detector_config),
            annotator=CharucoImageAnnotator.create(config.annotator_config),
        )

    def process_image(self, image: np.ndarray, annotate_image: bool = True) -> np.ndarray:
        self.observations.append(self.detector.detect(image))
        if annotate_image:
            return self.annotator.annotate_image(image, self.observations)
        return image


# def save_aruco_dictionary(dictionary: cv2.aruco.Dictionary,
#                           marker_size: int = 200,
#                           markers_per_row: int = 5,
#                           output_path: str = 'aruco_markers.png') -> None:
#     """
#     Save an image displaying each of the markers in the ArUco dictionary labeled by their ID.
#
#     :param dictionary: The ArUco dictionary to display.
#     :param marker_size: Size of each marker in pixels.
#     :param markers_per_row: Number of markers per row.
#     :param output_path: Path to save the output image.
#     """
#     num_markers = len(dictionary.bytesList)
#     rows = (num_markers + markers_per_row - 1) // markers_per_row
#     image_width = markers_per_row * marker_size
#     image_height = rows * marker_size
#
#     output_image = np.ones((image_height, image_width), dtype=np.uint8) * 255
#
#     for i in range(num_markers):
#         marker_id = i
#         marker_image = cv2.aruco.drawMarker(dictionary, marker_id, marker_size)
#         row = i // markers_per_row
#         col = i % markers_per_row
#         x = col * marker_size
#         y = row * marker_size
#         output_image[y:y + marker_size, x:x + marker_size] = marker_image
#
#         cv2.putText(output_image, str(marker_id), (x + 10, y + marker_size - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#
#     cv2.imwrite(output_path, output_image)

if __name__ == "__main__":
    # Aruco detection docs: https://docs.opencv.org/4.10.0/d5/dae/tutorial_aruco_detection.html
    CharucoTracker.create().demo()
