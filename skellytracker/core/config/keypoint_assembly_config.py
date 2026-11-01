"""Declarative selection of detector keypoints into a tracking output."""
from pydantic import BaseModel, ConfigDict, Field, model_validator


class KeypointSourceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    stage_path: tuple[str, ...] = Field(min_length=1)
    source_names: tuple[str, ...]
    selected_names: tuple[str, ...]

    @model_validator(mode="after")
    def validate_names(self):
        if len(set(self.source_names)) != len(self.source_names):
            raise ValueError("Source keypoint names must be unique")
        if not set(self.selected_names).issubset(self.source_names):
            raise ValueError("Selected keypoints must exist in the source schema")
        return self


class KeypointAssemblyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    target_stage: str
    sources: tuple[KeypointSourceConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sources(self):
        names = tuple(name for source in self.sources for name in source.selected_names)
        if len(set(names)) != len(names):
            raise ValueError("Assembled keypoint names must be unique")
        paths = tuple(source.stage_path for source in self.sources)
        if len(set(paths)) != len(paths):
            raise ValueError("Each source stage may occur only once")
        if any(path[0] != self.target_stage or len(path) > 2 for path in paths):
            raise ValueError("Assembly sources must be the target stage or its direct children")
        return self
