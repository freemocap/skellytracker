"""Keypoint-derived bounding box — a backend-agnostic ObjectDetector for
hierarchical crops (e.g. body → hand, body → face).

Unlike YOLOX or a precomputed detector, this ObjectDetector never looks at
the image: it derives a bounding box purely from a parent stage's keypoints
(e.g. wrist/index/pinky from a body detector), so it can crop a child
DetectionStage without running any additional model inference. Any
KeypointDetector can be plugged into that child stage — MediaPipe hands,
RTMPose hands, or anything else — since the crop it produces flows through
DetectionStage's normal BoundingBox/EMA-smoothing/coordinate-translation
pipeline like any other ObjectDetector's output.

Example — cropping a hand region from a body stage's wrist/index/pinky
keypoints::

    right_hand_bbox = KeypointBoundingBoxDetectorConfig(
        center_keypoint_names=("right_index", "right_pinky"),
        scale_keypoint_pairs=(("right_wrist", "right_index"), ("right_wrist", "right_pinky")),
        scale_factor=3.0,
    )

A body detector that only exposes wrist keypoints (no index/pinky, e.g.
RTMPose body) can instead size the crop off the elbow→wrist ("forearm")
length::

    right_hand_bbox = KeypointBoundingBoxDetectorConfig(
        center_keypoint_names=("right_wrist",),
        scale_keypoint_pairs=(("right_elbow", "right_wrist"),),
        scale_factor=1.5,
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.config.detector_configs import ObjectDetectorConfig
from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import (
    OBJECT_DETECTOR_REGISTRY,
    ObjectDetector,
)
from skellytracker.core.detectors.metadata import EmptyMetadata
from skellytracker.core.sessions.cpu_session import CpuSession
from skellytracker.core.sessions.session import Session


class KeypointBoundingBoxDetectorConfig(ObjectDetectorConfig):
    detector_type: Literal["keypoint_bbox"] = "keypoint_bbox"
    session_backend: Literal["cpu"] = "cpu"

    # Named points averaged (in image space) to get the box center. At least
    # one must be valid on a given frame or detect() returns no boxes.
    center_keypoint_names: tuple[str, ...]

    # Named point pairs whose pixel distance sizes the box; the largest valid
    # pairwise distance is used, so a partially-occluded frame still sizes
    # off whichever pair is visible. When none are valid, min_box_size_px
    # alone determines the size.
    scale_keypoint_pairs: tuple[tuple[str, str], ...] = ()

    # box side length = max(valid pairwise distance) * scale_factor
    scale_factor: float = 3.0
    min_box_size_px: float = 60.0
    min_visibility: float = 0.0


@dataclass
class KeypointBoundingBoxDetector(ObjectDetector):
    """Derives a square BoundingBox from named parent-stage keypoints."""

    config: KeypointBoundingBoxDetectorConfig = field(
        default_factory=lambda: KeypointBoundingBoxDetectorConfig(center_keypoint_names=())
    )
    session: Session = field(default_factory=CpuSession)

    def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], EmptyMetadata]:
        """Identity preprocess — returns image unchanged with empty metadata."""
        return image, EmptyMetadata()

    def postprocess(self, raw: Any, metadata: EmptyMetadata) -> list[BoundingBox]:
        """Identity postprocess — raw is already list[BoundingBox]."""
        return raw

    def detect(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext | None = None,
        parent_keypoints: Keypoints | None = None,
    ) -> list[BoundingBox]:
        if parent_keypoints is None:
            return []

        center = self._center(parent_keypoints)
        if center is None:
            return []
        cx, cy = center

        size = max(self._scale(parent_keypoints), self.config.min_box_size_px)
        return [BoundingBox.from_center_size(cx, cy, size, size)]

    def _center(self, keypoints: Keypoints) -> tuple[float, float] | None:
        points = []
        for name in self.config.center_keypoint_names:
            if not keypoints.has_name(name):
                continue
            idx = keypoints.index_of(name)
            if keypoints.visibility[idx] < self.config.min_visibility:
                continue
            xy = keypoints.xy_by_name(name)
            if np.isnan(xy).any():
                continue
            points.append(xy)
        if not points:
            return None
        mean = np.mean(points, axis=0)
        return float(mean[0]), float(mean[1])

    def _scale(self, keypoints: Keypoints) -> float:
        best = 0.0
        for name_a, name_b in self.config.scale_keypoint_pairs:
            if not (keypoints.has_name(name_a) and keypoints.has_name(name_b)):
                continue
            idx_a, idx_b = keypoints.index_of(name_a), keypoints.index_of(name_b)
            if (
                keypoints.visibility[idx_a] < self.config.min_visibility
                or keypoints.visibility[idx_b] < self.config.min_visibility
            ):
                continue
            xy_a, xy_b = keypoints.xy_by_name(name_a), keypoints.xy_by_name(name_b)
            if np.isnan(xy_a).any() or np.isnan(xy_b).any():
                continue
            distance = float(np.hypot(*(xy_a - xy_b)))
            best = max(best, distance * self.config.scale_factor)
        return best

    @classmethod
    def create(
        cls,
        config: ObjectDetectorConfig,
        session: Session,
    ) -> KeypointBoundingBoxDetector:
        return cls(config=config, session=session)


OBJECT_DETECTOR_REGISTRY["keypoint_bbox"] = KeypointBoundingBoxDetector
