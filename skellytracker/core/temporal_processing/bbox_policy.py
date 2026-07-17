from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.temporal_processing.temporal_processing_config import (
    BBoxAreaCollapseConfig,
    BBoxFitnessCheckConfig,
    BBoxPolicyConfig,
    KeypointsWithinBBoxRatioConfig,
    MaxFramesWithoutRedetectConfig,
    MinKeypointVisibilityConfig,
)
from skellytracker.core.tracker.tracker_state import StageState


@runtime_checkable
class BBoxFitnessCheck(Protocol):
    def fails(self, frame_number: int, stage_state: StageState) -> bool:
        ...


@dataclass
class MinKeypointVisibilityCheck:
    threshold: float

    def fails(self, frame_number: int, stage_state: StageState) -> bool:
        kpts = stage_state.last_keypoints
        if kpts is None or kpts.n_valid == 0:
            return False
        return float(kpts.visibility[kpts.valid_mask].mean()) < self.threshold


@dataclass
class KeypointsWithinBBoxRatioCheck:
    threshold: float

    def fails(self, frame_number: int, stage_state: StageState) -> bool:
        kpts = stage_state.last_keypoints
        bbox = stage_state.bbox_state.smooth_bbox
        if kpts is None or bbox is None or kpts.n_valid == 0:
            return False
        xy = kpts.xy[kpts.valid_mask]
        inside = (
            (xy[:, 0] >= bbox.x1) & (xy[:, 0] <= bbox.x2) &
            (xy[:, 1] >= bbox.y1) & (xy[:, 1] <= bbox.y2)
        )
        return float(inside.mean()) < self.threshold


@dataclass
class MaxFramesWithoutRedetectCheck:
    n_frames: int

    def fails(self, frame_number: int, stage_state: StageState) -> bool:
        last = stage_state.bbox_state.last_detection_frame
        if last is None:
            return True
        return (frame_number - last) >= self.n_frames


@dataclass
class BBoxAreaCollapseCheck:
    """Redetect when the keypoint-derived bbox for this frame is much smaller than the last known bbox."""

    min_area_ratio: float
    expansion_ratio: float
    min_visibility: float

    def fails(self, frame_number: int, stage_state: StageState) -> bool:
        smooth_bbox = stage_state.bbox_state.smooth_bbox
        kpts = stage_state.last_keypoints
        if smooth_bbox is None or kpts is None:
            return False
        predicted = predict_bbox_from_keypoints(kpts, self.expansion_ratio, self.min_visibility)
        if predicted is None:
            return True  # can't derive any bbox → track is lost
        expected_area = smooth_bbox.area
        if expected_area <= 0:
            return False
        return predicted.area < self.min_area_ratio * expected_area


def predict_bbox_from_keypoints(
    keypoints: Keypoints,
    expansion_ratio: float,
    min_visibility: float = 0.0,
) -> BoundingBox | None:
    """Compute a tight bbox around visible keypoints and expand it by expansion_ratio.

    Returns None if no valid keypoints are found.
    """
    if min_visibility > 0:
        mask = (keypoints.visibility >= min_visibility) & keypoints.valid_mask
    else:
        mask = keypoints.valid_mask

    if not mask.any():
        return None

    xy = keypoints.xyz[mask, :2]
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    if len(xy) == 0:
        return None

    x1, y1 = xy.min(axis=0)
    x2, y2 = xy.max(axis=0)

    if x2 <= x1 or y2 <= y1:
        return None

    w, h = x2 - x1, y2 - y1
    return BoundingBox(
        x1=x1 - w * expansion_ratio,
        y1=y1 - h * expansion_ratio,
        x2=x2 + w * expansion_ratio,
        y2=y2 + h * expansion_ratio,
        confidence=1.0,
    )


def _build_check(config: BBoxFitnessCheckConfig) -> BBoxFitnessCheck:
    if isinstance(config, MinKeypointVisibilityConfig):
        return MinKeypointVisibilityCheck(threshold=config.threshold)
    if isinstance(config, KeypointsWithinBBoxRatioConfig):
        return KeypointsWithinBBoxRatioCheck(threshold=config.threshold)
    if isinstance(config, MaxFramesWithoutRedetectConfig):
        return MaxFramesWithoutRedetectCheck(n_frames=config.n_frames)
    if isinstance(config, BBoxAreaCollapseConfig):
        return BBoxAreaCollapseCheck(
            min_area_ratio=config.min_area_ratio,
            expansion_ratio=config.expansion_ratio,
            min_visibility=config.min_visibility,
        )
    raise ValueError(f"Unknown fitness check config type: {type(config)}")


@dataclass
class BBoxPolicy:
    redetect_interval: int = 1
    fitness_checks: list[BBoxFitnessCheck] = field(default_factory=list)
    keypoint_bbox_expansion: float | None = None
    keypoint_bbox_min_visibility: float = 0.0

    def should_redetect(self, frame_number: int, stage_state: StageState) -> bool:
        last = stage_state.bbox_state.last_detection_frame
        if last is None:
            return True
        if (frame_number - last) >= self.redetect_interval:
            return True
        return any(check.fails(frame_number, stage_state) for check in self.fitness_checks)

    def predict_bbox(self, stage_state: StageState) -> BoundingBox | None:
        """Return a bbox to use when the object detector is being skipped.

        ``keypoint_tracked_bbox`` (tight-around-keypoints + one expansion) is
        recomputed every frame in DetectionStage.run from that frame's actual
        keypoints, regardless of whether the object detector ran. Here it is
        expanded a *second* time before being handed back as the next frame's
        crop — matching the old (skellytracker/old/rtmpose_tracker) two-stage
        update-then-predict expansion. Without this second expansion, keypoints
        sitting right at the tight box's edge (or ones with a one-frame
        confidence dip) fall outside the next crop and can never be
        re-acquired until the next scheduled redetect — the crop ratchets
        smaller frame over frame instead of tracking the subject.
        """
        tracked = stage_state.bbox_state.keypoint_tracked_bbox
        if self.keypoint_bbox_expansion is not None and tracked is not None:
            return tracked.scaled(1.0 + 2.0 * self.keypoint_bbox_expansion)
        return stage_state.bbox_state.smooth_bbox

    @classmethod
    def from_config(cls, config: BBoxPolicyConfig) -> BBoxPolicy:
        return cls(
            redetect_interval=config.redetect_interval,
            fitness_checks=[_build_check(c) for c in config.fitness_checks],
            keypoint_bbox_expansion=config.keypoint_bbox_expansion,
            keypoint_bbox_min_visibility=config.keypoint_bbox_min_visibility,
        )
