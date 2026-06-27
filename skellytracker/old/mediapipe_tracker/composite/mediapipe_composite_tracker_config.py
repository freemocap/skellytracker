from pydantic import Field

from skellytracker.old.base_tracker.base_tracker_abcs import BaseTrackerConfig
from skellytracker.old.mediapipe_tracker.composite.mediapipe_composite_annotator import \
    MediapipeCompositeAnnotatorConfig
from skellytracker.old.mediapipe_tracker.composite.mediapipe_composite_config import \
    MediapipeCompositeDetectorConfig


class MediapipeCompositeTrackerConfig(BaseTrackerConfig):
    detector_config: MediapipeCompositeDetectorConfig = Field(default_factory=MediapipeCompositeDetectorConfig)
    annotator_config: MediapipeCompositeAnnotatorConfig = Field(default_factory=MediapipeCompositeAnnotatorConfig)
