"""
Pipeline configuration types.

All detector/task configs live here.
Used to tell child processes which detector to instantiate.
"""
from functools import reduce
from operator import or_
from typing import Annotated, Any

from pydantic import Discriminator

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetectorConfig

CHARUCO_AVAILABLE = False
LEGACY_MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False
RTMPOSE_AVAILABLE = False

try:
    from skellytracker.trackers.charuco_tracker.charuco_detector import CharucoDetectorConfig
    CHARUCO_AVAILABLE = True
except ModuleNotFoundError:
    pass

try:
    from skellytracker.trackers.legacy_mediapipe_tracker.legacy_mediapipe_detector_config import LegacyMediapipeDetectorConfig
    LEGACY_MEDIAPIPE_AVAILABLE = True
except ModuleNotFoundError:
    pass

try:
    from skellytracker.trackers.mediapipe_tracker import MediapipeDetectorConfig
    MEDIAPIPE_AVAILABLE = True
except ModuleNotFoundError:
    pass

try:
    from skellytracker.trackers.rtmpose_tracker.rtmpose_detector import RTMPoseDetectorConfig
    RTMPOSE_AVAILABLE = True
except ModuleNotFoundError:
    pass


def _detect_detector_config_type(data: Any) -> str:
    """Inspect raw data to determine which detector config subclass to use."""
    if isinstance(data, BaseDetectorConfig):
        if CHARUCO_AVAILABLE and isinstance(data, CharucoDetectorConfig):
            return "charuco"
        if MEDIAPIPE_AVAILABLE and isinstance(data, MediapipeDetectorConfig):
            return "mediapipe"
        if LEGACY_MEDIAPIPE_AVAILABLE and isinstance(data, LegacyMediapipeDetectorConfig):
            return "legacy_mediapipe"
        if RTMPOSE_AVAILABLE and isinstance(data, RTMPoseDetectorConfig):
            return "rtmpose"
        raise ValueError(f"Unsupported detector config type: {type(data)}")

    if isinstance(data, dict):
        if "squares_x" in data or "aruco_dictionary_name" in data:
            return "charuco"
        if "pose_config" in data or "hand_config" in data or "face_config" in data:
            return "mediapipe"
        return "legacy_mediapipe"

    raise ValueError(f"Cannot determine detector config type from: {type(data)}")


_AVAILABLE_CONFIGS: list[type[BaseDetectorConfig]] = []

if CHARUCO_AVAILABLE:
    _AVAILABLE_CONFIGS.append(CharucoDetectorConfig)
if MEDIAPIPE_AVAILABLE:
    _AVAILABLE_CONFIGS.append(MediapipeDetectorConfig)
if LEGACY_MEDIAPIPE_AVAILABLE:
    _AVAILABLE_CONFIGS.append(LegacyMediapipeDetectorConfig)
if RTMPOSE_AVAILABLE:
    _AVAILABLE_CONFIGS.append(RTMPoseDetectorConfig)

if len(_AVAILABLE_CONFIGS) == 0:
    raise RuntimeError("No trackers available!")

# Build the Union type dynamically: ConfigA | ConfigB | ConfigC ...
_DetectorConfigUnion = reduce(or_, _AVAILABLE_CONFIGS)

DetectorConfig = Annotated[
    _DetectorConfigUnion,
    Discriminator(_detect_detector_config_type),
]
