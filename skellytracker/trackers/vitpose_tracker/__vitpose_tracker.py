from easy_ViTPose import VitInference
from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseTracker, BaseTrackerConfig, BaseRecorder
from skellytracker.trackers.vitpose_tracker.vitpose_annotator import VITPoseAnnotator, BaseImageAnnotatorConfig
from skellytracker.trackers.vitpose_tracker.vitpose_detector import VITPoseDetector, VITPoseDetectorConfig


class VITPoseTrackerConfig(BaseTrackerConfig):
    detector_config: VITPoseDetectorConfig = Field(default_factory=VITPoseDetectorConfig)
    annotator_config: BaseImageAnnotatorConfig = Field(default_factory=BaseImageAnnotatorConfig)


class VITPoseRecorder(BaseRecorder):
    pass


class VITPoseTracker(BaseTracker):
    config: VITPoseTrackerConfig
    detector: VITPoseDetector
    annotator: VITPoseAnnotator
    recorder: VITPoseRecorder | None = None

    @classmethod
    def create(cls, config: VITPoseTrackerConfig | None = None):
        if config is None:
            config = VITPoseTrackerConfig()

        detector = VITPoseDetector.create(config.detector_config)

        return cls(
            config=config,
            detector=detector,
            annotator=VITPoseAnnotator.create(config.annotator_config),
            recorder=VITPoseRecorder()
        )


if __name__ == "__main__":
    tracker = VITPoseTracker.create().demo()
