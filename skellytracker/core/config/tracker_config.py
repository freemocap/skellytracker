from __future__ import annotations

from pydantic import BaseModel

from skellytracker.core.config.detection_stage_config import DetectionStageConfig


class TrackerConfig(BaseModel):
    """Top-level declarative config for a Tracker.

    Pass to Tracker.create() alongside a sessions dict to build the full pipeline.
    """

    stages: list[DetectionStageConfig]
