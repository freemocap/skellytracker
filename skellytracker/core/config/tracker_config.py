from __future__ import annotations

from pydantic import BaseModel, model_validator

from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.config.keypoint_assembly_config import KeypointAssemblyConfig


class TrackerConfig(BaseModel):
    """Top-level declarative config for a Tracker.

    Pass to Tracker.create() alongside a sessions dict to build the full pipeline.
    """

    stages: list[DetectionStageConfig]
    preset_name: str | None = None
    keypoint_assembly: KeypointAssemblyConfig | None = None

    @model_validator(mode="after")
    def validate_assembly_stages(self):
        if self.keypoint_assembly is not None:
            for source in self.keypoint_assembly.sources:
                stages = self.stages
                for name in source.stage_path:
                    matches = [stage for stage in stages if stage.name == name]
                    if len(matches) != 1:
                        raise ValueError(f"Assembly stage path must resolve uniquely: {source.stage_path}")
                    stages = matches[0].children
        return self
