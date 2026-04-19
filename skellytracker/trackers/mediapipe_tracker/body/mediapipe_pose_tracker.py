import logging
from dataclasses import dataclass

from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseRecorder, BaseTracker, BaseTrackerConfig
from skellytracker.trackers.mediapipe_tracker.body.mediapipe_pose_annotator import (
    MediapipePoseAnnotator,
    MediapipePoseAnnotatorConfig,
)
from skellytracker.trackers.mediapipe_tracker.body.mediapipe_pose_config import MediapipePoseConfig
from skellytracker.trackers.mediapipe_tracker.body.mediapipe_pose_detector import MediapipePoseDetector

logger = logging.getLogger(__name__)


class MediapipePoseTrackerConfig(BaseTrackerConfig):
    detector_config: MediapipePoseConfig = Field(default_factory=MediapipePoseConfig)
    annotator_config: MediapipePoseAnnotatorConfig = Field(default_factory=MediapipePoseAnnotatorConfig)


class MediapipePoseRecorder(BaseRecorder):
    pass

@dataclass
class MediapipePoseTracker(BaseTracker):
    config: MediapipePoseTrackerConfig
    detector: MediapipePoseDetector
    annotator: MediapipePoseAnnotator
    recorder: MediapipePoseRecorder | None = None

    @classmethod
    def create(cls, config: MediapipePoseTrackerConfig | None = None) -> "MediapipePoseTracker":
        if config is None:
            config = MediapipePoseTrackerConfig()
        return cls(
            config=config,
            detector=MediapipePoseDetector.create(config=config.detector_config),
            annotator=MediapipePoseAnnotator.create(config=config.annotator_config),
            recorder=MediapipePoseRecorder(),
        )
