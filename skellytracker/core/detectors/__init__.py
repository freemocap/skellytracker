from skellytracker.core.detectors.detector_base_classes import (
    KEYPOINT_DETECTOR_REGISTRY,
    OBJECT_DETECTOR_REGISTRY,
    KeypointDetector,
    ObjectDetector,
    build_keypoint_detector,
    build_object_detector,
)
from skellytracker.core.detectors.object_detectors.precomputed import (
    PrecomputedObjectDetector,
    PrecomputedObjectDetectorConfig,
)

# Register concrete detector implementations here:
#
#   from skellytracker.core.detectors.object_detectors.yolox import YoloxPersonDetector
#   OBJECT_DETECTOR_REGISTRY["yolox_person"] = YoloxPersonDetector
#
#   from skellytracker.core.detectors.keypoint_detectors.rtmpose import RTMPoseKeypointDetector
#   KEYPOINT_DETECTOR_REGISTRY["rtmpose"] = RTMPoseKeypointDetector

__all__ = [
    "build_keypoint_detector",
    "build_object_detector",
    "KeypointDetector",
    "KEYPOINT_DETECTOR_REGISTRY",
    "ObjectDetector",
    "OBJECT_DETECTOR_REGISTRY",
    "PrecomputedObjectDetector",
    "PrecomputedObjectDetectorConfig",
]
