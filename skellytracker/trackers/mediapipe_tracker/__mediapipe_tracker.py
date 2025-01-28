import logging

import numpy as np
from pydantic import Field
from skellytracker.trackers.base_tracker.base_tracker import BaseTracker, BaseTrackerConfig
from skellytracker.trackers.mediapipe_tracker.mediapipe_annotator import MediapipeAnnotatorConfig, MediapipeImageAnnotator
from skellytracker.trackers.mediapipe_tracker.mediapipe_detector import MediapipeDetector, MediapipeDetectorConfig
from skellytracker.trackers.mediapipe_tracker.mediapipe_observation import MediapipeObservation, MediapipeResults

logger = logging.getLogger(__name__)

class MediapipeTrackerConfig(BaseTrackerConfig):
    detector_config: MediapipeDetectorConfig = Field(default_factory = MediapipeDetectorConfig)
    annotator_config: MediapipeAnnotatorConfig = Field(default_factory = MediapipeAnnotatorConfig)


class MediapipeTracker(BaseTracker):
    config: MediapipeTrackerConfig
    detector: MediapipeDetector
    annotator: MediapipeImageAnnotator

    @classmethod
    def create(cls, config: MediapipeTrackerConfig | None = None):
        if config is None:
            config = MediapipeTrackerConfig()
        detector = MediapipeDetector.create(config.detector_config)

        return cls(
            config=config,
            detector=detector,
            annotator=MediapipeImageAnnotator.create(config.annotator_config),

        )

    def process_image(self,
                      image: np.ndarray,
                      annotate_image: bool = False) -> tuple[np.ndarray, MediapipeObservation, MediapipeResults] | tuple[MediapipeObservation, MediapipeResults]:

        latest_observation, mediapipe_results = self.detector.detect(image)
        if annotate_image:
            return self.annotator.annotate_image(image=image,
                                                 latest_observation=latest_observation), latest_observation, mediapipe_results

        return latest_observation, mediapipe_results


if __name__ == "__main__":
    MediapipeTracker.create().demo()
