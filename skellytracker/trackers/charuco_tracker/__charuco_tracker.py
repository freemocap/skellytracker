# Charuco Detector docs https://docs.opencv.org/4.10.0/d9/df5/classcv_1_1aruco_1_1CharucoDetector.html
# Aruco detection docs: https://docs.opencv.org/4.10.0/d5/dae/tutorial_aruco_detection.html

from dataclasses import field, dataclass

import numpy as np

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker, BaseTrackerConfig
from skellytracker.trackers.charuco_tracker.charuco_annotator import CharucoAnnotatorConfig, CharucoImageAnnotator
from skellytracker.trackers.charuco_tracker.charuco_camera_calibrator import CameraCalibrationEstimate
from skellytracker.trackers.charuco_tracker.charuco_detector import CharucoDetectorConfig, CharucoDetector
from skellytracker.trackers.charuco_tracker.charuco_observations import CharucoObservations, CharucoObservation


class CharucoTrackerConfig(BaseTrackerConfig):
    detector_config: CharucoDetectorConfig
    annotator_config: CharucoAnnotatorConfig

    @classmethod
    def create(cls):
        return cls(detector_config=CharucoDetectorConfig(),
                   annotator_config=CharucoAnnotatorConfig())


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




if __name__ == "__main__":
    CharucoTracker.create().demo()
