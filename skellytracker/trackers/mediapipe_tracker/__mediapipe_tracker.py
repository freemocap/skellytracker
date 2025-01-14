import logging
from dataclasses import dataclass, field

import numpy as np

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker, BaseTrackerConfig, BaseRecorder
from skellytracker.trackers.mediapipe_tracker.mediapipe_annotator import MediapipeAnnotatorConfig, MediapipeImageAnnotator
from skellytracker.trackers.mediapipe_tracker.mediapipe_detector import MediapipeDetector, MediapipeDetectorConfig
from skellytracker.trackers.mediapipe_tracker.mediapipe_observation import MediapipeObservation, MediapipeResults

logger = logging.getLogger(__name__)

@dataclass
class MediapipeTrackerConfig(BaseTrackerConfig):
    detector_config: MediapipeDetectorConfig = field(default_factory = MediapipeDetectorConfig)
    annotator_config: MediapipeAnnotatorConfig = field(default_factory = MediapipeAnnotatorConfig)

@dataclass
class MediapipeRecorder(BaseRecorder):
    # TODO: the BaseRecorder covers most of this, but we could save metadata with this if we wanted
    pass


@dataclass
class MediapipeTracker(BaseTracker):
    config: MediapipeTrackerConfig
    detector: MediapipeDetector
    annotator: MediapipeImageAnnotator
    recorder: MediapipeRecorder

    @classmethod
    def create(cls, config: MediapipeTrackerConfig | None = None):
        if config is None:
            config = MediapipeTrackerConfig()
        detector = MediapipeDetector.create(config.detector_config)

        return cls(
            config=config,
            detector=detector,
            annotator=MediapipeImageAnnotator.create(config.annotator_config),
            recorder=MediapipeRecorder(),
        )

    def process_image(self,
                      image: np.ndarray,
                      annotate_image: bool = False,
                      record_observation: bool = True) -> tuple[np.ndarray, MediapipeObservation, MediapipeResults] | tuple[MediapipeObservation, MediapipeResults]:

        latest_observation, mediapipe_results = self.detector.detect(image)
        if record_observation:
            self.recorder.add_observations(observation=latest_observation)
        if annotate_image:
            return self.annotator.annotate_image(image=image,
                                                 latest_observation=latest_observation), latest_observation, mediapipe_results

        return latest_observation, mediapipe_results


if __name__ == "__main__":
    MediapipeTracker.create().demo()
