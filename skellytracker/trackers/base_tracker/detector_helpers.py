"""
Factory functions for creating detectors and annotators from config objects.

Called inside child processes — detector/annotator class imports are deferred
to avoid pulling in mediapipe/cv2.aruco at module import time. Config classes
are lightweight Pydantic models and safe to import eagerly (when available).
"""
from functools import reduce
from operator import or_
from typing import Annotated

from pydantic import Discriminator

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


# ============================================================================
# Discriminated union type for use as a Pydantic field annotation.
#
# Each config subclass has a `tracker_type: Literal["..."]` field with a
# unique default value. Pydantic uses this field to determine which subclass
# to deserialize into — no callable discriminator or Tag annotations needed.
# ============================================================================

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

_DetectorConfigUnion = reduce(or_, _AVAILABLE_CONFIGS)

SkeletonDetectorConfig = Annotated[
    _DetectorConfigUnion,
    Discriminator("tracker_type"),
]
