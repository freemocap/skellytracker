from skellytracker.core.detectors.detector_base_classes import (
    KEYPOINT_DETECTOR_REGISTRY,
    OBJECT_DETECTOR_REGISTRY,
    KeypointDetector,
    ObjectDetector,
    build_keypoint_detector,
    build_object_detector,
)

# Register concrete detector implementations here:
#
#   from skellytracker.core.detectors.yolo_detector import YOLODetector
#   OBJECT_DETECTOR_REGISTRY["yolo"] = YOLODetector
#
#   from skellytracker.core.detectors.rtmpose_detector import RTMPoseDetector
#   KEYPOINT_DETECTOR_REGISTRY["rtmpose"] = RTMPoseDetector

__all__ = [
    "build_keypoint_detector",
    "build_object_detector",
    "KeypointDetector",
    "KEYPOINT_DETECTOR_REGISTRY",
    "ObjectDetector",
    "OBJECT_DETECTOR_REGISTRY",
]
