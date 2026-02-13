from pydantic import Field
from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseTrackerConfig
from skellytracker.trackers.charuco_tracker.charuco_annotator import CharucoAnnotatorConfig
from skellytracker.trackers.charuco_tracker.charuco_detector import CharucoDetectorConfig


class CharucoTrackerConfig(BaseTrackerConfig):
    detector_config: CharucoDetectorConfig = Field(default_factory = CharucoDetectorConfig)
    annotator_config: CharucoAnnotatorConfig = Field(default_factory = CharucoAnnotatorConfig)
