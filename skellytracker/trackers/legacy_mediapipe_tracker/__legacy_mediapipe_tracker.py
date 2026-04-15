from dataclasses import dataclass
import logging

from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseTracker, BaseTrackerConfig, BaseRecorder
from skellytracker.trackers.legacy_mediapipe_tracker.legacy_mediapipe_annotator import LegacyMediapipeAnnotatorConfig, \
    LegacyMediapipeImageAnnotator
from skellytracker.trackers.legacy_mediapipe_tracker.legacy_mediapipe_detector import LegacyMediapipeDetector
from skellytracker.trackers.legacy_mediapipe_tracker.legacy_mediapipe_detector_config import \
    LegacyMediapipeDetectorConfig

logger = logging.getLogger(__name__)

class LegacyMediapipeTrackerConfig(BaseTrackerConfig):
    detector_config: LegacyMediapipeDetectorConfig = Field(default_factory = LegacyMediapipeDetectorConfig)
    annotator_config: LegacyMediapipeAnnotatorConfig = Field(default_factory = LegacyMediapipeAnnotatorConfig)

class LegacyMediapipeRecorder(BaseRecorder):
    # TODO: the BaseRecorder covers most of this, but we could save metadata with this if we wanted
    pass

@dataclass
class LegacyMediapipeTracker(BaseTracker):
    config: LegacyMediapipeTrackerConfig
    detector: LegacyMediapipeDetector
    annotator: LegacyMediapipeImageAnnotator | None = None
    recorder: LegacyMediapipeRecorder | None = None

    @classmethod
    def create(cls, config: LegacyMediapipeTrackerConfig | None = None):
        if config is None:
            config = LegacyMediapipeTrackerConfig()
        detector = LegacyMediapipeDetector.create(config.detector_config)

        return cls(
            config=config,
            detector=detector,
            annotator=LegacyMediapipeImageAnnotator.create(config.annotator_config),
            recorder=LegacyMediapipeRecorder(),
        )


if __name__ == "__main__":
    LegacyMediapipeTracker.create().demo()
