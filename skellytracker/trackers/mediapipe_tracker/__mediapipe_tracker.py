import logging

import numpy as np
from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker, BaseTrackerConfig, BaseRecorder
from skellytracker.trackers.mediapipe_tracker.mediapipe_annotator import MediapipeAnnotatorConfig, MediapipeImageAnnotator
from skellytracker.trackers.mediapipe_tracker.mediapipe_detector import MediapipeDetector, MediapipeDetectorConfig
from skellytracker.trackers.mediapipe_tracker.mediapipe_observation import MediapipeObservation, MediapipeResults

logger = logging.getLogger(__name__)

class MediapipeTrackerConfig(BaseTrackerConfig):
    detector_config: MediapipeDetectorConfig = Field(default_factory = MediapipeDetectorConfig)
    annotator_config: MediapipeAnnotatorConfig = Field(default_factory = MediapipeAnnotatorConfig)

class MediapipeRecorder(BaseRecorder):
    # TODO: the BaseRecorder covers most of this, but we could save metadata with this if we wanted
    pass


class MediapipeTracker(BaseTracker):
    config: MediapipeTrackerConfig
    detector: MediapipeDetector
    annotator: MediapipeImageAnnotator | None = None
    recorder: MediapipeRecorder | None = None

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
                        frame_number: int,
                        image: np.ndarray,
                        record_observation: bool = True) -> MediapipeObservation:

        latest_observation = self.detector.detect(image=image, frame_number=frame_number)

        if record_observation and self.recorder is not None:
            self.recorder.add_observation(observation=latest_observation)

        return latest_observation


if __name__ == "__main__":
    MediapipeTracker.create().demo()
