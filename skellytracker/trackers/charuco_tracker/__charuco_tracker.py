# Charuco Detector docs https://docs.opencv.org/4.10.0/d9/df5/classcv_1_1aruco_1_1CharucoDetector.html
# Aruco detection docs: https://docs.opencv.org/4.10.0/d5/dae/tutorial_aruco_detection.html

from dataclasses import dataclass, field

import numpy as np


from skellytracker.trackers.base_tracker.base_tracker import BaseTracker, BaseTrackerConfig
from skellytracker.trackers.charuco_tracker.camera_calibration_estimation import CameraCalibrationEstimator
from skellytracker.trackers.charuco_tracker.charuco_observation import CharucoObservation
from skellytracker.trackers.charuco_tracker.charuco_annotator import CharucoAnnotatorConfig, CharucoImageAnnotator
from skellytracker.trackers.charuco_tracker.charuco_detector import CharucoDetectorConfig, CharucoDetector


@dataclass
class CharucoTrackerConfig(BaseTrackerConfig):
    detector_config: CharucoDetectorConfig = field(default_factory = CharucoDetectorConfig)
    annotator_config: CharucoAnnotatorConfig = field(default_factory = CharucoAnnotatorConfig)


@dataclass
class CharucoTracker(BaseTracker):
    config: CharucoTrackerConfig
    detector: CharucoDetector
    annotator: CharucoImageAnnotator
    camera_calibration_estimator: CameraCalibrationEstimator | None = None

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

    def process_image(self,
                      image: np.ndarray,
                      annotate_image: bool = False) -> tuple[np.ndarray, CharucoObservation] | CharucoObservation:
        if self.camera_calibration_estimator is None:
            self.camera_calibration_estimator = CameraCalibrationEstimator.create_initial(image_size=image.shape[:2],
                                                                                          aruco_marker_ids=list(self.detector.board.getIds()),
                                                                                          aruco_corners_in_object_coordinates=self.detector.board.getObjPoints(), # type: ignore
                                                                                          charuco_corner_ids=self.config.detector_config.charuco_corner_ids,
                                                                                          charuco_corners_in_object_coordinates=self.detector.board.getChessboardCorners(), # type: ignore
                                                                                          )
        latest_observation = self.detector.detect(image)
        self.camera_calibration_estimator.add_observation(latest_observation)
        if annotate_image:
            return self.annotator.annotate_image(image=image,
                                                 latest_observation=latest_observation), latest_observation

        return latest_observation


if __name__ == "__main__":
    CharucoTracker.create().demo()
