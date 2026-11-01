"""Prepared dense keypoint assembly shared by single-image and batch tracking."""
from dataclasses import dataclass

import numpy as np

from skellytracker.core.config.keypoint_assembly_config import KeypointAssemblyConfig
from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.data_primitives.observation import Observation, StageObservation


@dataclass(frozen=True)
class PreparedKeypointAssembly:
    config: KeypointAssemblyConfig
    template: Keypoints
    selections: tuple[tuple[np.ndarray, slice], ...]

    @classmethod
    def prepare(cls, config: KeypointAssemblyConfig):
        names = tuple(name for source in config.sources for name in source.selected_names)
        selections = []
        start = 0
        for source in config.sources:
            lookup = {name: index for index, name in enumerate(source.source_names)}
            indices = np.asarray([lookup[name] for name in source.selected_names], dtype=np.intp)
            indices.flags.writeable = False
            stop = start + len(indices)
            selections.append((indices, slice(start, stop)))
            start = stop
        return cls(config, Keypoints.empty(names), tuple(selections))

    def apply(self, observation: Observation) -> None:
        target = observation.stages.get(self.config.target_stage)
        if target is None:
            target = StageObservation(name=self.config.target_stage)
        # Fresh values are required: shared workers and recordings retain observations.
        # Only schema metadata/selection indices are reused across cameras and frames.
        xyz = np.full_like(self.template.xyz, np.nan)
        visibility = np.zeros_like(self.template.visibility)
        for source, (indices, destination) in zip(self.config.sources, self.selections):
            stage = target if len(source.stage_path) == 1 else target.children.get(source.stage_path[1])
            points = None if stage is None else stage.keypoints
            if points is None:
                continue
            if points.names != source.source_names:
                raise ValueError(f"Keypoint schema mismatch in {source.stage_path}")
            np.take(points.xyz, indices, axis=0, out=xyz[destination])
            np.take(points.visibility, indices, out=visibility[destination])
        target.keypoints = self.template.with_values(xyz=xyz, visibility=visibility)
        for source in self.config.sources:
            if len(source.stage_path) == 2:
                child = target.children.get(source.stage_path[1])
                if child is not None:
                    # Keep crop geometry and provenance, but publish numerical
                    # keypoints only once through the assembled parent tensor.
                    child.keypoints = None
        observation.stages[self.config.target_stage] = target
