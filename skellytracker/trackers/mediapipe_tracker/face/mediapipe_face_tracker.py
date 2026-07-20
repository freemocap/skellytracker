import logging
from dataclasses import dataclass

from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseRecorder, BaseTracker, BaseTrackerConfig
from skellytracker.trackers.mediapipe_tracker.face.mediapipe_face_annotator import (
    MediapipeFaceAnnotator,
    MediapipeFaceAnnotatorConfig,
)
from skellytracker.trackers.mediapipe_tracker.face.mediapipe_face_config import MediapipeFaceConfig
from skellytracker.trackers.mediapipe_tracker.face.mediapipe_face_detector import MediapipeFaceDetector

logger = logging.getLogger(__name__)


class MediapipeFaceTrackerConfig(BaseTrackerConfig):
    detector_config: MediapipeFaceConfig = Field(default_factory=MediapipeFaceConfig)
    annotator_config: MediapipeFaceAnnotatorConfig = Field(default_factory=MediapipeFaceAnnotatorConfig)


class MediapipeFaceRecorder(BaseRecorder):
    pass

@dataclass
class MediapipeFaceTracker(BaseTracker):
    config: MediapipeFaceTrackerConfig
    detector: MediapipeFaceDetector
    annotator: MediapipeFaceAnnotator
    recorder: MediapipeFaceRecorder | None = None

    @classmethod
    def create(cls, config: MediapipeFaceTrackerConfig | None = None) -> "MediapipeFaceTracker":
        if config is None:
            config = MediapipeFaceTrackerConfig()
        return cls(
            config=config,
            detector=MediapipeFaceDetector.create(config=config.detector_config),
            annotator=MediapipeFaceAnnotator.create(config=config.annotator_config),
            recorder=MediapipeFaceRecorder(),
        )
