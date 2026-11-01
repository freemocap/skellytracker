from __future__ import annotations

from pydantic import BaseModel, SerializeAsAny, field_validator

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
    object_detector: SerializeAsAny[ObjectDetectorConfig] | None = None
    keypoint_detectors: list[SerializeAsAny[KeypointDetectorConfig]] = []
    children: list[DetectionStageConfig] = []
    bbox_policy: BBoxPolicyConfig = BBoxPolicyConfig()
    bbox_smoothing: BBoxSmoothingConfig | None = None
    keypoint_smoothing: KeypointSmoothingConfig | KalmanKeypointSmoothingConfig | None = None
    keypoint_reset_policy: KeypointResetPolicyConfig = KeypointResetPolicyConfig()

    @field_validator("object_detector", mode="before")
    @classmethod
    def restore_object_config(cls, value):
        return _restore_detector_config(value, keypoint=False)

    @field_validator("keypoint_detectors", mode="before")
    @classmethod
    def restore_keypoint_configs(cls, values):
        return [_restore_detector_config(value, keypoint=True) for value in values]


def _restore_detector_config(value, *, keypoint: bool):
    """Restore concrete config types through the same registry used for construction."""
    if not isinstance(value, dict):
        return value
    # Register built-ins without constructing sessions or loading model assets.
    import skellytracker.core.detectors.keypoint_detectors.aruco
    import skellytracker.core.detectors.keypoint_detectors.charuco
    import skellytracker.core.detectors.keypoint_detectors.mediapipe
    import skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.rtmw_wholebody_detector
    import skellytracker.core.detectors.keypoint_detectors.rtmpose
    import skellytracker.core.detectors.object_detectors.yolox
    from typing import get_type_hints
    from skellytracker.core.detectors.detector_base_classes import (
        KEYPOINT_DETECTOR_REGISTRY, OBJECT_DETECTOR_REGISTRY,
    )
    registry = KEYPOINT_DETECTOR_REGISTRY if keypoint else OBJECT_DETECTOR_REGISTRY
    detector_type = value.get("detector_type")
    if detector_type not in registry:
        raise ValueError(f"Unregistered detector configuration {detector_type!r}")
    config_type = get_type_hints(registry[detector_type])["config"]
    return config_type.model_validate(value)


DetectionStageConfig.model_rebuild()
