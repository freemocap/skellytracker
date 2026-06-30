from __future__ import annotations

from pydantic import BaseModel

from skellytracker.core.config.detector_configs import (
    KeypointDetectorConfig,
    ObjectDetectorConfig,
)


class DetectionStageConfig(BaseModel):
    """Declarative description of a DetectionStage and its subtree."""

    name: str
    object_detector: ObjectDetectorConfig | None = None
    keypoint_detectors: list[KeypointDetectorConfig] = []
    children: list[DetectionStageConfig] = []


DetectionStageConfig.model_rebuild()
