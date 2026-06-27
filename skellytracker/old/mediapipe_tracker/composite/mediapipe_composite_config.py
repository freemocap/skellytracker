from typing import Literal

from pydantic import Field

from skellytracker.old.base_tracker.base_tracker_abcs import BaseDetectorConfig, TrackerType
from skellytracker.old.mediapipe_tracker.body.mediapipe_pose_config import MediapipePoseConfig
from skellytracker.old.mediapipe_tracker.face.mediapipe_face_config import MediapipeFaceConfig
from skellytracker.old.mediapipe_tracker.hands.mediapipe_hand_config import MediapipeHandConfig


class MediapipeCompositeDetectorConfig(BaseDetectorConfig):
    tracker_type: Literal[TrackerType.MEDIAPIPE] = TrackerType.MEDIAPIPE
    pose_config: MediapipePoseConfig = Field(default_factory=MediapipePoseConfig)
    hand_config: MediapipeHandConfig = Field(default_factory=MediapipeHandConfig)
    face_config: MediapipeFaceConfig = Field(default_factory=MediapipeFaceConfig)

    confidence_threshold: float = 0.5

    # ROI crop parameters
    hand_roi_scale: float = 2.0
    face_roi_scale: float = 2.5
    roi_visibility_threshold: float = 0.5  # below this, fall back to full-image detection
    roi_smoothing: float = 0.5  # EMA smoothing for ROI boxes (0.0 = no smoothing, higher = more smoothing)
    hand_bbox_padding: float = 1.8  # multiplier on previous frame's hand bbox diagonal for crop sizing
    min_hand_crop_image_fraction: float = 0.15  # minimum hand crop as fraction of image height (cold start fallback)
    hand_overlap_threshold: float = 0.3  # if detected hand wrists are closer than this fraction of hand bbox diagonal, treat as duplicates

    # Which sub-detectors to run
    detect_hands: bool = True
    detect_face: bool = True
