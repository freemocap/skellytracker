import logging

from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker, BaseTrackerConfig, BaseRecorder
from skellytracker.trackers.bright_point_tracker.brightest_point_annotator import BrightestPointAnnotatorConfig, BrightestPointImageAnnotator
from skellytracker.trackers.bright_point_tracker.brightest_point_detector import BrightestPointDetector, BrightestPointDetectorConfig

logger = logging.getLogger(__name__)

class BrightestPointTrackerConfig(BaseTrackerConfig):
    detector_config: BrightestPointDetectorConfig = Field(default_factory = BrightestPointDetectorConfig)
    annotator_config: BrightestPointAnnotatorConfig = Field(default_factory = BrightestPointAnnotatorConfig)

class BrightestPointRecorder(BaseRecorder):
    pass


class BrightestPointTracker(BaseTracker):
    config: BrightestPointTrackerConfig
    detector: BrightestPointDetector
    annotator: BrightestPointImageAnnotator | None = None
    recorder: BrightestPointRecorder | None = None

    @classmethod
    def create(cls, config: BrightestPointTrackerConfig | None = None):
        if config is None:
            config = BrightestPointTrackerConfig()
        detector = BrightestPointDetector.create(config.detector_config)

        return cls(
            config=config,
            detector=detector,
            annotator=BrightestPointImageAnnotator.create(config.annotator_config),
            recorder=BrightestPointRecorder(),
        )


if __name__ == "__main__":
    BrightestPointTracker.create().demo()
