from skellytracker.core.detectors.keypoint_detectors.mediapipe.face import mediapipe_face_detector
from skellytracker.core.detectors.keypoint_detectors.mediapipe.hands import mediapipe_hand_detector
from skellytracker.core.detectors.keypoint_detectors.mediapipe.body import mediapipe_pose_detector
from skellytracker.core.detectors.keypoint_detectors.mediapipe.face.mediapipe_face_detector import (
    MediapipeFaceDetectorConfig,
    MediapipeFaceKeypointDetector,
)
from skellytracker.core.detectors.keypoint_detectors.mediapipe.hands.mediapipe_hand_detector import (
    MediapipeHandDetectorConfig,
    MediapipeHandKeypointDetector,
)
from skellytracker.core.detectors.keypoint_detectors.mediapipe.mediapipe_model_manager import (
    MediapipePoseModelComplexity,
)
from skellytracker.core.detectors.keypoint_detectors.mediapipe.body.mediapipe_pose_detector import (
    MediapipePoseDetectorConfig,
    MediapipePoseKeypointDetector,
)
from skellytracker.core.sessions.mediapipe_session import (
    MediaPipeSession,
    MediaPipeSessionConfig,
)

__all__ = [
    "MediapipeFaceDetectorConfig",
    "MediapipeFaceKeypointDetector",
    "MediapipeHandDetectorConfig",
    "MediapipeHandKeypointDetector",
    "MediapipePoseDetectorConfig",
    "MediapipePoseKeypointDetector",
    "MediapipePoseModelComplexity",
    "MediaPipeSession",
    "MediaPipeSessionConfig",
    # Import detector modules to trigger registry side-effects
    "mediapipe_face_detector",
    "mediapipe_hand_detector",
    "mediapipe_pose_detector",
]
