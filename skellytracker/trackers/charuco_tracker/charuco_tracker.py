from typing import Dict, List

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker, BaseTrackerConfig, BaseObservation, \
    TrackedPointId, BaseDetectorConfig, BaseImageAnnotatorConfig, BaseDetector, BaseRecorder

DEFAULT_ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

class CharucoObservation(BaseObservation):
    charuco_ids: List[int]
    charuco_corners: np.ndarray
    marker_ids: List[int]
    marker_corners: np.ndarray

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
    def aruco_marker_ids(self) -> List[int]:
        return list(range(self.squares_x * self.squares_y))

    @property
    def charuco_corner_names(self) -> List[str]:
        return [f"CharucoCorner_{index}" for index in self.charuco_corner_ids]

    @property
    def aruco_marker_names(self) -> List[str]:
        return [f"ArucoMarker_{index}" for index in self.aruco_marker_ids]


class CharucoObservationFactory(BaseModel):
    corner_ids: List[int]
    marker_ids: List[int]

    @classmethod
    def create(cls, config: CharucoDetectorConfig):
        return cls(
            corner_ids=config.charuco_corner_ids,
            marker_ids=config.aruco_marker_ids,
        )

    def create_observation(
            self,
            charuco_corner_ids: List[int],
            charuco_corners_in: np.ndarray,
            aruco_marker_ids: List[int],
            aruco_marker_corners_in: np.ndarray
    ) -> CharucoObservation:
        charuco_corners_out = np.ndarray((len(self.corner_ids), 2), np.float32)
        charuco_corners_out[:] = np.nan
        marker_corners_out = np.ndarray((len(self.marker_ids), 2), np.float32)
        marker_corners_out[:] = np.nan

        for corner_id, corner in zip(charuco_corner_ids, charuco_corners_in):
            charuco_corners_out[corner_id] = corner

        for marker_id, corner in zip(aruco_marker_ids, aruco_marker_corners_in):
            marker_corners_out[marker_id] = corner

        return CharucoObservation(
            charuco_ids=self.corner_ids,
            charuco_corners=charuco_corners_out,
            marker_ids=self.marker_ids,
            marker_corners=marker_corners_out,
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

class CharucoImageAnnotator(BaseModel):
    config: CharucoAnnotatorConfig

    @classmethod
    def create(cls, config: CharucoAnnotatorConfig):
        return cls(config=config)

    def annotate_image(
            self, image: np.ndarray, observations: CharucoObservations,
    ) -> np.ndarray:
        # Copy the original image for annotation
        annotated_image = image.copy()

        # Draw a marker for each tracked corner
        for observation in observations:
            if (
                    observation.pixel_x is not None
                    and observation.pixel_y is not None
            ):
                cv2.drawMarker(
                    annotated_image,
                    (int(observation.pixel_x), int(observation.pixel_y)),
                    self.config.marker_color,
                    markerType=self.config.marker_type,
                    markerSize=self.config.marker_size,
                    thickness=self.config.marker_thickness,
                )
                cv2.putText(
                    annotated_image,
                    observation.id,
                    (int(observation.pixel_x), int(observation.pixel_y)),
                    self.config.text_font,
                    self.config.text_size,
                    self.config.text_color,
                    self.config.text_thickness,
                )

        return annotated_image

class CharucoDetector(BaseDetector):
    detector_config: CharucoDetectorConfig
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

        return cls(
            detector_config=config,
            detector=cv2.aruco.CharucoDetector(board),
            observation_factory=CharucoObservationFactory.create(config),
        )

    def detect(self, image: np.ndarray) -> CharucoObservation:
        if len(image.shape) == 2:
            grey_image = image
        else:
            grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.observation_factory.create_observation(*self.detector.detectBoard(grey_image))




class CharucoTracker(BaseTracker):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: CharucoTrackerConfig
    detector: CharucoDetector
    annotator: CharucoImageAnnotator
    observations: CharucoObservations = []

    @classmethod
    def create(cls, config: CharucoTrackerConfig):
        return cls(
            config=config,
            detector=CharucoDetector.create(config.detector_config),
            annotator=CharucoImageAnnotator.create(config.annotator_config),
        )


    def process_image(self, image: np.ndarray, annotate_image: bool = True) -> np.ndarray|None:
        self.observations.append(self.detector.detect(image))
        if annotate_image:
            return self.annotator.annotate_image(image, self.observations)




if __name__ == "__main__":
    CharucoTracker().demo()
