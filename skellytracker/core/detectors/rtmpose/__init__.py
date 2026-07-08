import skellytracker.core.detectors.rtmpose.wholebody.rtmpose_wholebody_detector  # noqa: F401 (registry)
import skellytracker.core.detectors.rtmpose.body.rtmpose_body_detector  # noqa: F401 (registry)
import skellytracker.core.detectors.rtmpose.hand.rtmpose_hand_detector  # noqa: F401 (registry)
import skellytracker.core.detectors.rtmpose.face.rtmpose_face_detector  # noqa: F401 (registry)

from skellytracker.core.detectors.rtmpose.wholebody.rtmpose_wholebody_detector import (
    RTMPoseDetectorConfig,
    RTMPoseKeypointDetector,
    RTMPOSE_MODEL_SPECS,
)
from skellytracker.core.detectors.rtmpose.body.rtmpose_body_detector import (
    RTMPoseBodyDetectorConfig,
    RTMPoseBodyDetector,
    RTMPOSE_BODY_MODEL_SPECS,
)
from skellytracker.core.detectors.rtmpose.hand.rtmpose_hand_detector import (
    RTMPoseHandDetectorConfig,
    RTMPoseHandDetector,
    RTMPOSE_HAND_MODEL_SPECS,
)
from skellytracker.core.detectors.rtmpose.face.rtmpose_face_detector import (
    RTMPoseFaceDetectorConfig,
    RTMPoseFaceDetector,
    RTMPOSE_FACE_MODEL_SPECS,
)

__all__ = [
    # Wholebody (backward-compat names match existing public API)
    "RTMPoseDetectorConfig",
    "RTMPoseKeypointDetector",
    "RTMPOSE_MODEL_SPECS",
    # Body
    "RTMPoseBodyDetectorConfig",
    "RTMPoseBodyDetector",
    "RTMPOSE_BODY_MODEL_SPECS",
    # Hand
    "RTMPoseHandDetectorConfig",
    "RTMPoseHandDetector",
    "RTMPOSE_HAND_MODEL_SPECS",
    # Face
    "RTMPoseFaceDetectorConfig",
    "RTMPoseFaceDetector",
    "RTMPOSE_FACE_MODEL_SPECS",
]
