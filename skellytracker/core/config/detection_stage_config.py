from __future__ import annotations

from pydantic import BaseModel

from skellytracker.core.config.detector_configs import (
    KeypointDetectorConfig,
    ObjectDetectorConfig,
)
from skellytracker.core.temporal_processing.temporal_processing_config import (
    BBoxPolicyConfig,
    BBoxSmoothingConfig,
    KalmanKeypointSmoothingConfig,
    KeypointResetPolicyConfig,
    KeypointSmoothingConfig,
)


class DetectionStageConfig(BaseModel):
    """Declarative description of a DetectionStage and its subtree."""

    name: str
    object_detector: ObjectDetectorConfig | None = None
    keypoint_detectors: list[KeypointDetectorConfig] = []
    children: list[DetectionStageConfig] = []
    bbox_policy: BBoxPolicyConfig = BBoxPolicyConfig()
    bbox_smoothing: BBoxSmoothingConfig | None = None
    keypoint_smoothing: KeypointSmoothingConfig | KalmanKeypointSmoothingConfig | None = None
    keypoint_reset_policy: KeypointResetPolicyConfig = KeypointResetPolicyConfig()


DetectionStageConfig.model_rebuild()
