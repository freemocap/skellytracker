from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from skellytracker.core.annotation.annotator import Annotator
from skellytracker.core.observation import Observation, StageObservation
from skellytracker.core.data_primitives import Keypoints


@dataclass
class StageAnnotationSchema:
    """Visual schema for one stage: connections between named points and color."""
    connections: tuple[tuple[str, str], ...]  # name pairs from YAML
    keypoint_color: tuple[int, int, int] = (0, 255, 0)
    connection_color: tuple[int, int, int] = (0, 200, 0)
    keypoint_radius: int = 4
    connection_thickness: int = 2


@dataclass
class KeypointAnnotatorConfig:
    stage_schemas: dict[str, StageAnnotationSchema] = field(default_factory=dict)
    default_keypoint_color: tuple[int, int, int] = (0, 255, 0)
    default_connection_color: tuple[int, int, int] = (0, 200, 0)
    keypoint_radius: int = 4
    connection_thickness: int = 2
    confidence_threshold: float = 0.0


@dataclass
class KeypointAnnotator(Annotator):
    """Draws keypoints and skeleton connections from an Observation.

    Works with any Observation; no tracker-specific knowledge required.
    Per-stage visual schemas (colors, connections) are supplied at construction.
    Stages without a schema still have their keypoints drawn using defaults.
    """

    config: KeypointAnnotatorConfig

    def annotate(
        self,
        image: NDArray[np.uint8],
        observation: Observation,
    ) -> NDArray[np.uint8]:
        out = image.copy()
        self._annotate_stages(out, observation.stages)
        return out

    def _annotate_stages(
        self,
        image: NDArray[np.uint8],
        stages: dict[str, StageObservation],
    ) -> None:
        for stage_obs in stages.values():
            if stage_obs.keypoints is not None:
                schema = self.config.stage_schemas.get(stage_obs.name)
                self._draw_stage(image, stage_obs.keypoints, schema)
            if stage_obs.children:
                self._annotate_stages(image, stage_obs.children)

    def _draw_stage(
        self,
        image: NDArray[np.uint8],
        keypoints: Keypoints,
        schema: StageAnnotationSchema | None,
    ) -> None:
        threshold = self.config.confidence_threshold

        if schema is not None:
            kp_color = schema.keypoint_color
            conn_color = schema.connection_color
            radius = schema.keypoint_radius
            thickness = schema.connection_thickness
            connections = schema.connections
        else:
            kp_color = self.config.default_keypoint_color
            conn_color = self.config.default_connection_color
            radius = self.config.keypoint_radius
            thickness = self.config.connection_thickness
            connections = ()

        # Draw connections first so keypoints render on top
        for name_a, name_b in connections:
            if not (keypoints.has_name(name_a) and keypoints.has_name(name_b)):
                continue
            if keypoints.visibility[keypoints.index_of(name_a)] < threshold:
                continue
            if keypoints.visibility[keypoints.index_of(name_b)] < threshold:
                continue
            pt_a = keypoints.xy_by_name(name_a)
            pt_b = keypoints.xy_by_name(name_b)
            if np.isnan(pt_a).any() or np.isnan(pt_b).any():
                continue
            cv2.line(
                image,
                (int(pt_a[0]), int(pt_a[1])),
                (int(pt_b[0]), int(pt_b[1])),
                conn_color,
                thickness,
            )

        # Draw keypoints
        for i, _name in enumerate(keypoints.names):
            if keypoints.visibility[i] < threshold:
                continue
            pt = keypoints.xyz[i, :2]
            if np.isnan(pt).any():
                continue
            cv2.circle(image, (int(pt[0]), int(pt[1])), radius, kp_color, -1)

    @classmethod
    def create(cls, config: object) -> KeypointAnnotator:
        if not isinstance(config, KeypointAnnotatorConfig):
            raise TypeError(f"Expected KeypointAnnotatorConfig, got {type(config).__name__}")
        return cls(config=config)
