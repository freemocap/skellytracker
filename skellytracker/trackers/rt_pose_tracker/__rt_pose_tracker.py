from dataclasses import dataclass, field

from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseRecorder, BaseTracker, BaseTrackerConfig
from skellytracker.trackers.rt_pose_tracker.rt_pose_annotator import RtPoseAnnotator, RtPoseAnnotatorConfig
from skellytracker.trackers.rt_pose_tracker.rt_pose_detector import RtPoseDetector, RtPoseDetectorConfig
from skellytracker.trackers.rt_pose_tracker.rt_pose_observation import RtPoseObservation


class RtPoseTrackerConfig(BaseTrackerConfig):
    detector_config: RtPoseDetectorConfig = Field(default_factory=RtPoseDetectorConfig)
    annotator_config: RtPoseAnnotatorConfig = Field(default_factory=RtPoseAnnotatorConfig)


@dataclass
class RtPoseRecorder(BaseRecorder):
    observations: list[RtPoseObservation] = field(default_factory=list)


class RtPoseTracker(BaseTracker):
    config: RtPoseTrackerConfig
    detector: RtPoseDetector
    annotator: RtPoseAnnotator
    recorder: RtPoseRecorder | None = None

    @classmethod
    def create(cls, config: RtPoseTrackerConfig | None = None) -> "RtPoseTracker":
        if config is None:
            config = RtPoseTrackerConfig()

        return cls(
            config=config,
            detector=RtPoseDetector.create(config.detector_config),
            annotator=RtPoseAnnotator.create(config.annotator_config),
            recorder=RtPoseRecorder(),
        )


if __name__ == "__main__":
    RtPoseTracker.create().demo()
