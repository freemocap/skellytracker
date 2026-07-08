# Re-exports for backward compatibility. Import from the submodule directly
# for new code: skellytracker.core.detectors.rtmpose.wholebody
from skellytracker.core.detectors.rtmpose.wholebody.rtmpose_wholebody_detector import (
    RTMPoseDetectorConfig,
    RTMPoseKeypointDetector,
    RTMPOSE_MODEL_SPECS,
)

__all__ = [
    "RTMPoseDetectorConfig",
    "RTMPoseKeypointDetector",
    "RTMPOSE_MODEL_SPECS",
]
