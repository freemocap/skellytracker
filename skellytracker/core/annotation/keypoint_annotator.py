from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from skellytracker.core.annotation.annotator import Annotator
from skellytracker.core.data_primitives import BoundingBox, Keypoints
from skellytracker.core.data_primitives.observation import Observation, StageObservation


@dataclass
class ConnectionGroupSchema:
    """A named subset of connections drawn in a single color."""
    connections: tuple[tuple[str, str], ...]
    connection_color: tuple[int, int, int]
    connection_thickness: int = 2
    keypoint_color: tuple[int, int, int] | None = None  # None = fall back to stage default


@dataclass
class StageAnnotationSchema:
    """Visual schema for one stage: connections between named points and color."""
    connections: tuple[tuple[str, str], ...] = ()  # name pairs from YAML
    keypoint_color: tuple[int, int, int] = (0, 255, 0)
    connection_color: tuple[int, int, int] = (0, 200, 0)
    keypoint_radius: int = 4
    connection_thickness: int = 2
    draw_boxes: bool = False
    box_color_detected: tuple[int, int, int] = (0, 200, 0)     # detector actually ran this frame
    box_color_reused: tuple[int, int, int] = (0, 140, 255)     # bbox reused/predicted, detector skipped
    box_thickness: int = 2
    connection_groups: tuple[ConnectionGroupSchema, ...] = ()


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
            schema = self.config.stage_schemas.get(stage_obs.name)
            if schema is not None and schema.draw_boxes and stage_obs.bounding_boxes:
                self._draw_boxes(image, stage_obs.bounding_boxes, schema, stage_obs.detector_ran)
            if stage_obs.keypoints is not None:
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

        # Build per-keypoint color map from groups (only when groups define keypoint_color).
        kp_color_map: dict[str, tuple[int, int, int]] = {}
        if schema is not None and schema.connection_groups:
            for group in schema.connection_groups:
                if group.keypoint_color is not None:
                    for name_a, name_b in group.connections:
                        kp_color_map.setdefault(name_a, group.keypoint_color)
                        kp_color_map.setdefault(name_b, group.keypoint_color)

        # Draw connections first so keypoints render on top.
        # When connection_groups are defined, use per-group colors; otherwise fall back to the
        # flat connections list with a single connection_color.
        if schema is not None and schema.connection_groups:
            for group in schema.connection_groups:
                for name_a, name_b in group.connections:
                    self._draw_connection(image, keypoints, name_a, name_b, group.connection_color, group.connection_thickness, threshold)
        else:
            for name_a, name_b in connections:
                self._draw_connection(image, keypoints, name_a, name_b, conn_color, thickness, threshold)

        # Draw keypoints
        for i, name in enumerate(keypoints.names):
            if keypoints.visibility[i] < threshold:
                continue
            pt = keypoints.xyz[i, :2]
            if np.isnan(pt).any():
                continue
            color = kp_color_map.get(name, kp_color)
            cv2.circle(image, (int(pt[0]), int(pt[1])), radius, color, -1)

    def _draw_connection(
        self,
        image: NDArray[np.uint8],
        keypoints: Keypoints,
        name_a: str,
        name_b: str,
        color: tuple[int, int, int],
        thickness: int,
        threshold: float,
    ) -> None:
        if not (keypoints.has_name(name_a) and keypoints.has_name(name_b)):
            return
        if keypoints.visibility[keypoints.index_of(name_a)] < threshold:
            return
        if keypoints.visibility[keypoints.index_of(name_b)] < threshold:
            return
        pt_a = keypoints.xy_by_name(name_a)
        pt_b = keypoints.xy_by_name(name_b)
        if np.isnan(pt_a).any() or np.isnan(pt_b).any():
            return
        cv2.line(image, (int(pt_a[0]), int(pt_a[1])), (int(pt_b[0]), int(pt_b[1])), color, thickness)

    def _draw_boxes(
        self,
        image: NDArray[np.uint8],
        boxes: list[BoundingBox],
        schema: StageAnnotationSchema,
        detector_ran: bool,
    ) -> None:
        color = schema.box_color_detected if detector_ran else schema.box_color_reused
        for box in boxes:
            x1, y1 = int(box.x1), int(box.y1)
            x2, y2 = int(box.x2), int(box.y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, schema.box_thickness)
            label = f"{box.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    @classmethod
    def create(cls, config: object) -> KeypointAnnotator:
        if not isinstance(config, KeypointAnnotatorConfig):
            raise TypeError(f"Expected KeypointAnnotatorConfig, got {type(config).__name__}")
        return cls(config=config)
