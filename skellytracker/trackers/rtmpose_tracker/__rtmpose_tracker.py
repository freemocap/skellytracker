from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseImageAnnotatorConfig,
    BaseRecorder,
    BaseTracker,
    BaseTrackerConfig,
)
from skellytracker.trackers.rtmpose_tracker.rtmpose_annotator import (
    RTMPoseImageAnnotator,
)
from skellytracker.trackers.rtmpose_tracker.rtmpose_detector import (
    RTMPoseDetector,
    RTMPoseDetectorConfig,
)


class RTMPoseTrackerConfig(BaseTrackerConfig):
    detector_config: RTMPoseDetectorConfig = Field(
        default_factory=RTMPoseDetectorConfig
    )
    annotator_config: BaseImageAnnotatorConfig = Field(
        default_factory=BaseImageAnnotatorConfig
    )


class RTMPoseRecorder(BaseRecorder):
    pass


class RTMPoseTracker(BaseTracker):
    config: RTMPoseTrackerConfig
    detector: RTMPoseDetector
    annotator: RTMPoseImageAnnotator
    recorder: RTMPoseRecorder | None = None

    @classmethod
    def create(
        cls,
        config: RTMPoseTrackerConfig | None = None,
    ) -> "RTMPoseTracker":
        if config is None:
            config = RTMPoseTrackerConfig()

        detector = RTMPoseDetector.create(
            config.detector_config
        )

        annotator = RTMPoseImageAnnotator.create(
            config.annotator_config
        )

        return cls(
            config=config,
            detector=detector,
            annotator=annotator,
            recorder=RTMPoseRecorder(),
        )


if __name__ == "__main__":
    RTMPoseTracker.create().demo()