"""Backward-compat re-export.

The RTMW wholebody detector moved to
`skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.rtmw_wholebody_detector`
(specs/sidecar-implementation-plan.md M4). Import from there for new code —
this module only exists so existing imports of `RTMPoseDetectorConfig`/
`RTMPoseKeypointDetector`/`RTMPOSE_MODEL_SPECS` from this path keep working.
"""

from skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.rtmw_wholebody_detector import (
    RTMW_WHOLEBODY_MODEL_SPECS as RTMPOSE_MODEL_SPECS,
)
from skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.rtmw_wholebody_detector import (
    RTMWWholebodyDetector as RTMPoseKeypointDetector,
)
from skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.rtmw_wholebody_detector import (
    RTMWWholebodyDetectorConfig as RTMPoseDetectorConfig,
)

__all__ = ["RTMPoseDetectorConfig", "RTMPoseKeypointDetector", "RTMPOSE_MODEL_SPECS"]
