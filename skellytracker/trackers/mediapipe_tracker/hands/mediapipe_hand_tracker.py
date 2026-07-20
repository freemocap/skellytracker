import logging
from dataclasses import dataclass

from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseRecorder, BaseTracker, BaseTrackerConfig
from skellytracker.trackers.mediapipe_tracker.hands.mediapipe_hand_annotator import (
    MediapipeHandAnnotator,
    MediapipeHandAnnotatorConfig,
)
from skellytracker.trackers.mediapipe_tracker.hands.mediapipe_hand_config import MediapipeHandConfig
from skellytracker.trackers.mediapipe_tracker.hands.mediapipe_hand_detector import MediapipeHandDetector

logger = logging.getLogger(__name__)


class MediapipeHandTrackerConfig(BaseTrackerConfig):
    detector_config: MediapipeHandConfig = Field(default_factory=MediapipeHandConfig)
    annotator_config: MediapipeHandAnnotatorConfig = Field(default_factory=MediapipeHandAnnotatorConfig)


class MediapipeHandRecorder(BaseRecorder):
    pass

@dataclass
class MediapipeHandTracker(BaseTracker):
    config: MediapipeHandTrackerConfig
    detector: MediapipeHandDetector
    annotator: MediapipeHandAnnotator
    recorder: MediapipeHandRecorder | None = None

    @classmethod
    def create(cls, config: MediapipeHandTrackerConfig | None = None) -> "MediapipeHandTracker":
        if config is None:
            config = MediapipeHandTrackerConfig()
        return cls(
            config=config,
            detector=MediapipeHandDetector.create(config=config.detector_config),
            annotator=MediapipeHandAnnotator.create(config=config.annotator_config),
            recorder=MediapipeHandRecorder(),
        )
