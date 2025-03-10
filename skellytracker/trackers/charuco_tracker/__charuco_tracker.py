# Charuco Detector docs https://docs.opencv.org/4.10.0/d9/df5/classcv_1_1aruco_1_1CharucoDetector.html
# Aruco detection docs: https://docs.opencv.org/4.10.0/d5/dae/tutorial_aruco_detection.html

import logging

import numpy as np
from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker, BaseTrackerConfig
from skellytracker.trackers.charuco_tracker.charuco_annotator import CharucoAnnotatorConfig, CharucoImageAnnotator
from skellytracker.trackers.charuco_tracker.charuco_detector import CharucoDetectorConfig, CharucoDetector
from skellytracker.trackers.charuco_tracker.charuco_observation import CharucoObservation

logger = logging.getLogger(__name__)

class CharucoTrackerConfig(BaseTrackerConfig):
    detector_config: CharucoDetectorConfig = Field(default_factory = CharucoDetectorConfig)
    annotator_config: CharucoAnnotatorConfig = Field(default_factory = CharucoAnnotatorConfig)


class CharucoTracker(BaseTracker):
    config: CharucoTrackerConfig
    detector: CharucoDetector
    annotator: CharucoImageAnnotator

    @classmethod
    def create(cls, config: CharucoTrackerConfig | None = None):
        if config is None:
            config = CharucoTrackerConfig()
        detector = CharucoDetector.create(config.detector_config)

        return cls(
            config=config,
            detector=detector,
            annotator=CharucoImageAnnotator.create(config.annotator_config),

        )

    @property
    def  aruco_corners_in_object_coordinates(self):
        return self.detector.board.getObjPoints()  # type: ignore

    @property
    def charuco_corner_ids(self ):
        return self.config.detector_config.charuco_corner_ids

    @property
    def charuco_corners_in_object_coordinates(self):
        return self.detector.board.getChessboardCorners()


if __name__ == "__main__":
    CharucoTracker.create().demo()
