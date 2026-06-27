"""
Factory functions for creating detectors and annotators from config objects,
plus the discriminated-union type aliases used by downstream consumers to
declare "I want any skeleton tracker" / "I want any board tracker" / etc.

Called inside child processes — detector/annotator class imports are deferred
to avoid pulling in mediapipe/cv2.aruco at module import time. Config classes
are lightweight Pydantic models and safe to import eagerly (when available).
"""
from functools import reduce
from operator import or_
from typing import Annotated

from pydantic import Discriminator

from skellytracker.old.base_tracker.base_tracker_abcs import (
    BaseDetector,
    BaseDetectorConfig,
    BaseImageAnnotator,
)

CHARUCO_AVAILABLE = False
LEGACY_MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False
RTMPOSE_AVAILABLE = False
VITPOSE_AVAILABLE = False
BRIGHTEST_POINT_AVAILABLE = False

try:
    from skellytracker.old.charuco_tracker.charuco_tracker_config import CharucoDetectorConfig
    CHARUCO_AVAILABLE = True
except ModuleNotFoundError:
    pass

try:
    from skellytracker.old.legacy_mediapipe_tracker import LegacyMediapipeDetectorConfig
    LEGACY_MEDIAPIPE_AVAILABLE = True
except ModuleNotFoundError:
    pass

try:
    from skellytracker.old.mediapipe_tracker import MediapipeDetectorConfig
    MEDIAPIPE_AVAILABLE = True
except ModuleNotFoundError:
    pass

try:
    from skellytracker.old.rtmpose_tracker.rtmpose_detector_config import RTMPoseDetectorConfig
    RTMPOSE_AVAILABLE = True
except ModuleNotFoundError:
    pass

try:
    from skellytracker.old.vitpose_tracker.vitpose_detector import VITPoseDetectorConfig
    VITPOSE_AVAILABLE = True
except ModuleNotFoundError:
    pass

try:
    from skellytracker.old.brightest_point_tracker.brightest_point_detector import BrightestPointDetectorConfig
    BRIGHTEST_POINT_AVAILABLE = True
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
        from skellytracker.old.charuco_tracker.charuco_detector import CharucoDetector
        return CharucoDetector.create(config=detector_config)

    if MEDIAPIPE_AVAILABLE and isinstance(detector_config, MediapipeDetectorConfig):
        from skellytracker.old.mediapipe_tracker import MediapipeDetector
        return MediapipeDetector.create(config=detector_config)

    if LEGACY_MEDIAPIPE_AVAILABLE and isinstance(detector_config, LegacyMediapipeDetectorConfig):
        from skellytracker.old.legacy_mediapipe_tracker.legacy_mediapipe_detector import LegacyMediapipeDetector
        return LegacyMediapipeDetector.create(config=detector_config)

    if RTMPOSE_AVAILABLE and isinstance(detector_config, RTMPoseDetectorConfig):
        from skellytracker.old.rtmpose_tracker.rtmpose_detector import RTMPoseDetector
        return RTMPoseDetector.create(config=detector_config)

    if VITPOSE_AVAILABLE and isinstance(detector_config, VITPoseDetectorConfig):
        from skellytracker.old.vitpose_tracker.vitpose_detector import VITPoseDetector
        return VITPoseDetector.create(config=detector_config)

    if BRIGHTEST_POINT_AVAILABLE and isinstance(detector_config, BrightestPointDetectorConfig):
        from skellytracker.old.brightest_point_tracker.brightest_point_detector import BrightestPointDetector
        return BrightestPointDetector.create(config=detector_config)

    raise TypeError(
        f"No detector available for config type: {type(detector_config).__name__}. "
        f"Available trackers — charuco: {CHARUCO_AVAILABLE}, mediapipe: {MEDIAPIPE_AVAILABLE}, "
        f"legacy_mediapipe: {LEGACY_MEDIAPIPE_AVAILABLE}, rtmpose: {RTMPOSE_AVAILABLE}, "
        f"vitpose: {VITPOSE_AVAILABLE}, brightest_point: {BRIGHTEST_POINT_AVAILABLE}"
    )


def create_annotator_from_config(config: BaseDetectorConfig) -> BaseImageAnnotator:
    """
    Create an image annotator matching the given detector config.

    Called inside child processes. Same deferred-import rationale as
    create_detector_from_config.
    """
    if CHARUCO_AVAILABLE and isinstance(config, CharucoDetectorConfig):
        from skellytracker.old.charuco_tracker.charuco_annotator import CharucoImageAnnotator, CharucoAnnotatorConfig
        return CharucoImageAnnotator.create(config=CharucoAnnotatorConfig())

    if MEDIAPIPE_AVAILABLE and isinstance(config, MediapipeDetectorConfig):
        from skellytracker.old.mediapipe_tracker import MediapipeAnnotator, MediapipeAnnotatorConfig
        return MediapipeAnnotator.create(config=MediapipeAnnotatorConfig())

    if LEGACY_MEDIAPIPE_AVAILABLE and isinstance(config, LegacyMediapipeDetectorConfig):
        from skellytracker.old.legacy_mediapipe_tracker.legacy_mediapipe_annotator import LegacyMediapipeImageAnnotator, LegacyMediapipeAnnotatorConfig
        return LegacyMediapipeImageAnnotator.create(config=LegacyMediapipeAnnotatorConfig())

    if RTMPOSE_AVAILABLE and isinstance(config, RTMPoseDetectorConfig):
        raise NotImplementedError("RTMPose annotator not yet implemented")

    if VITPOSE_AVAILABLE and isinstance(config, VITPoseDetectorConfig):
        from skellytracker.old.vitpose_tracker.vitpose_annotator import VITPoseAnnotator
        return VITPoseAnnotator.create()

    if BRIGHTEST_POINT_AVAILABLE and isinstance(config, BrightestPointDetectorConfig):
        from skellytracker.old.brightest_point_tracker.brightest_point_annotator import BrightestPointImageAnnotator, BrightestPointAnnotatorConfig
        return BrightestPointImageAnnotator.create(config=BrightestPointAnnotatorConfig())

    raise TypeError(
        f"No annotator available for config type: {type(config).__name__}. "
        f"Available trackers — charuco: {CHARUCO_AVAILABLE}, mediapipe: {MEDIAPIPE_AVAILABLE}, "
        f"legacy_mediapipe: {LEGACY_MEDIAPIPE_AVAILABLE}, rtmpose: {RTMPOSE_AVAILABLE}, "
        f"vitpose: {VITPOSE_AVAILABLE}, brightest_point: {BRIGHTEST_POINT_AVAILABLE}"
    )


# ============================================================================
# Discriminated-union type aliases for use as Pydantic field annotations.
#
# Each config subclass has a `tracker_type: Literal[TrackerType.X]` field with
# a unique enum-member value. Pydantic uses that field to route deserialization
# to the correct subclass. Categories map 1:1 onto the physical nature of what
# the tracker tracks (skeleton/pose, calibration board, generic point).
#
# Each alias is built only from members whose optional deps imported above —
# so a partial install still yields a valid (narrower) union. If no members
# in a category are available the alias is `None` (downstream code can check
# for that). Only fails loudly when *no* tracker is available at all.
# ============================================================================

_SKELETON_CONFIGS: list[type[BaseDetectorConfig]] = []
if MEDIAPIPE_AVAILABLE:
    _SKELETON_CONFIGS.append(MediapipeDetectorConfig)
if LEGACY_MEDIAPIPE_AVAILABLE:
    _SKELETON_CONFIGS.append(LegacyMediapipeDetectorConfig)
if RTMPOSE_AVAILABLE:
    _SKELETON_CONFIGS.append(RTMPoseDetectorConfig)
if VITPOSE_AVAILABLE:
    _SKELETON_CONFIGS.append(VITPoseDetectorConfig)

_BOARD_CONFIGS: list[type[BaseDetectorConfig]] = []
if CHARUCO_AVAILABLE:
    _BOARD_CONFIGS.append(CharucoDetectorConfig)

_POINT_CONFIGS: list[type[BaseDetectorConfig]] = []
if BRIGHTEST_POINT_AVAILABLE:
    _POINT_CONFIGS.append(BrightestPointDetectorConfig)

if not (_SKELETON_CONFIGS or _BOARD_CONFIGS or _POINT_CONFIGS):
    raise RuntimeError("No trackers available!")


def _build_discriminated_union(configs: list[type[BaseDetectorConfig]]):
    if not configs:
        return None
    return Annotated[reduce(or_, configs), Discriminator("tracker_type")]


SkeletonDetectorConfig = _build_discriminated_union(_SKELETON_CONFIGS)
BoardDetectorConfig = _build_discriminated_union(_BOARD_CONFIGS)
PointDetectorConfig = _build_discriminated_union(_POINT_CONFIGS)
