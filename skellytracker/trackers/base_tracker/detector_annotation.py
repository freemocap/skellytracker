"""
Pipeline configuration types.

All detector/task configs live here.
used to tell child processes which detector to instantiate.
"""
from typing import Annotated, Any, Union

from pydantic import Discriminator, Tag
from skellytracker.trackers.charuco_tracker.charuco_detector import CharucoDetectorConfig
from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetectorConfig
from skellytracker.trackers.mediapipe_tracker import MediapipeDetectorConfig
from skellytracker.trackers.legacy_mediapipe_tracker.legacy_mediapipe_detector_config import \
    LegacyMediapipeDetectorConfig


try:
    from skellytracker.trackers.rtmpose_tracker.rtmpose_detector import RTMPoseDetectorConfig
except ModuleNotFoundError:
    RTMPoseDetectorConfig = BaseDetectorConfig


# TODO - This should live in skellytracker
def _detect_detector_config_type(data: Any) -> str:
    """Inspect raw data to determine which detector config subclass to use."""
    if isinstance(data, BaseDetectorConfig):
        if isinstance(data, CharucoDetectorConfig):
            return "charuco"
        if isinstance(data, MediapipeDetectorConfig):
            return "mediapipe"
        return "legacy_mediapipe"
    if isinstance(data, dict):
        if "squares_x" in data or "aruco_dictionary_name" in data:
            return "charuco"
        if "pose_config" in data or "hand_config" in data or "face_config" in data:
            return "mediapipe"
    return "legacy_mediapipe"


# TODO - move this to skellytracker
DetectorConfig = Annotated[
    Union[
        Annotated[LegacyMediapipeDetectorConfig, Tag("legacy_mediapipe")],
        Annotated[MediapipeDetectorConfig, Tag("mediapipe")],
        Annotated[CharucoDetectorConfig, Tag("charuco")],
        # Annotated[RTMPoseDetectorConfig, Tag("rtmpose")],
    ],
    Discriminator(_detect_detector_config_type),
]
