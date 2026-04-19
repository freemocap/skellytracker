"""
Factory functions for creating detectors and annotators from config objects.

Called inside child processes — detector/annotator class imports are deferred
to avoid pulling in mediapipe/cv2.aruco at module import time. Config classes
are lightweight Pydantic models and safe to import eagerly (when available).
"""
from functools import reduce
from operator import or_
from typing import Annotated

from pydantic import Discriminator, Tag

from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseDetector,
    BaseDetectorConfig,
    BaseImageAnnotator,
)

CHARUCO_AVAILABLE = False
LEGACY_MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False
RTMPOSE_AVAILABLE = False

try:
    from skellytracker.trackers.charuco_tracker.charuco_tracker_config import CharucoDetectorConfig
    CHARUCO_AVAILABLE = True
except ModuleNotFoundError:
    pass

try:
    from skellytracker.trackers.legacy_mediapipe_tracker import LegacyMediapipeDetectorConfig
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


def create_detector_from_config(detector_config: BaseDetectorConfig) -> BaseDetector:
    """
    Create a detector instance from a picklable config.

    Called inside child processes. The heavy detector class imports (mediapipe,
    cv2.aruco, etc.) are deferred to function scope intentionally to avoid
    loading them at module import time in the parent process.
    """
    if CHARUCO_AVAILABLE and isinstance(detector_config, CharucoDetectorConfig):
        from skellytracker.trackers.charuco_tracker.charuco_detector import CharucoDetector
        return CharucoDetector.create(config=detector_config)

    if MEDIAPIPE_AVAILABLE and isinstance(detector_config, MediapipeDetectorConfig):
        from skellytracker.trackers.mediapipe_tracker import MediapipeDetector
        return MediapipeDetector.create(config=detector_config)

    if LEGACY_MEDIAPIPE_AVAILABLE and isinstance(detector_config, LegacyMediapipeDetectorConfig):
        from skellytracker.trackers.legacy_mediapipe_tracker.legacy_mediapipe_detector import LegacyMediapipeDetector
        return LegacyMediapipeDetector.create(config=detector_config)

    if RTMPOSE_AVAILABLE and isinstance(detector_config, RTMPoseDetectorConfig):
        from skellytracker.trackers.rtmpose_tracker.rtmpose_detector import RTMPoseDetector
        return RTMPoseDetector.create(config=detector_config)

    raise TypeError(
        f"No detector available for config type: {type(detector_config).__name__}. "
        f"Available trackers — charuco: {CHARUCO_AVAILABLE}, mediapipe: {MEDIAPIPE_AVAILABLE}, "
        f"legacy_mediapipe: {LEGACY_MEDIAPIPE_AVAILABLE}, rtmpose: {RTMPOSE_AVAILABLE}"
    )


def create_annotator_from_config(config: BaseDetectorConfig) -> BaseImageAnnotator:
    """
    Create an image annotator matching the given detector config.

    Called inside child processes. Same deferred-import rationale as
    create_detector_from_config.
    """
    if CHARUCO_AVAILABLE and isinstance(config, CharucoDetectorConfig):
        from skellytracker.trackers.charuco_tracker.charuco_annotator import CharucoImageAnnotator, CharucoAnnotatorConfig
        return CharucoImageAnnotator.create(config=CharucoAnnotatorConfig())

    if MEDIAPIPE_AVAILABLE and isinstance(config, MediapipeDetectorConfig):
        from skellytracker.trackers.mediapipe_tracker import MediapipeAnnotator, MediapipeAnnotatorConfig
        return MediapipeAnnotator.create(config=MediapipeAnnotatorConfig())

    if LEGACY_MEDIAPIPE_AVAILABLE and isinstance(config, LegacyMediapipeDetectorConfig):
        from skellytracker.trackers.legacy_mediapipe_tracker.legacy_mediapipe_annotator import LegacyMediapipeImageAnnotator, LegacyMediapipeAnnotatorConfig
        return LegacyMediapipeImageAnnotator.create(config=LegacyMediapipeAnnotatorConfig())

    if RTMPOSE_AVAILABLE and isinstance(config, RTMPoseDetectorConfig):
        raise NotImplementedError("RTMPose annotator not yet implemented")

    raise TypeError(
        f"No annotator available for config type: {type(config).__name__}. "
        f"Available trackers — charuco: {CHARUCO_AVAILABLE}, mediapipe: {MEDIAPIPE_AVAILABLE}, "
        f"legacy_mediapipe: {LEGACY_MEDIAPIPE_AVAILABLE}, rtmpose: {RTMPOSE_AVAILABLE}"
    )

def _detect_detector_config_type(data: object) -> str:
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


# Each union member must be Annotated[ConfigType, Tag("...")] where the tag
# string matches what _detect_detector_config_type returns for that type.
_TAGGED_CONFIGS: list[type] = []

if CHARUCO_AVAILABLE:
    _TAGGED_CONFIGS.append(Annotated[CharucoDetectorConfig, Tag("charuco")])
if MEDIAPIPE_AVAILABLE:
    _TAGGED_CONFIGS.append(Annotated[MediapipeDetectorConfig, Tag("mediapipe")])
if LEGACY_MEDIAPIPE_AVAILABLE:
    _TAGGED_CONFIGS.append(Annotated[LegacyMediapipeDetectorConfig, Tag("legacy_mediapipe")])
if RTMPOSE_AVAILABLE:
    _TAGGED_CONFIGS.append(Annotated[RTMPoseDetectorConfig, Tag("rtmpose")])

if len(_TAGGED_CONFIGS) == 0:
    raise RuntimeError("No trackers available!")

# Build the Union type dynamically: TaggedA | TaggedB | TaggedC ...
_DetectorConfigUnion = reduce(or_, _TAGGED_CONFIGS)

SkeletonDetectorConfig = Annotated[
    _DetectorConfigUnion,
    Discriminator(_detect_detector_config_type),
]
