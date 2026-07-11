from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class MinKeypointVisibilityConfig(BaseModel):
    kind: Literal["min_keypoint_visibility"] = "min_keypoint_visibility"
    threshold: float = 0.3


class KeypointsWithinBBoxRatioConfig(BaseModel):
    kind: Literal["keypoints_within_bbox_ratio"] = "keypoints_within_bbox_ratio"
    threshold: float = 0.5


class MaxFramesWithoutRedetectConfig(BaseModel):
    kind: Literal["max_frames_without_redetect"] = "max_frames_without_redetect"
    n_frames: int = 30


class BBoxAreaCollapseConfig(BaseModel):
    """Redetect when the keypoint-derived bbox has shrunk drastically vs the last known bbox.

    Catches cases where the person has left the frame or the track has switched.
    Uses the same keypoint-to-bbox logic as keypoint_bbox_expansion.
    """

    kind: Literal["bbox_area_collapse"] = "bbox_area_collapse"
    min_area_ratio: float = 0.25
    expansion_ratio: float = 0.05
    min_visibility: float = 0.0


BBoxFitnessCheckConfig = Annotated[
    MinKeypointVisibilityConfig
    | KeypointsWithinBBoxRatioConfig
    | MaxFramesWithoutRedetectConfig
    | BBoxAreaCollapseConfig,
    Field(discriminator="kind"),
]


class BBoxPolicyConfig(BaseModel):
    """Controls when the ObjectDetector re-runs and how to fill the bbox when it doesn't.

    When the detector is skipped, the bbox for the current frame can either be
    re-derived from the previous frame's keypoints (if keypoint_bbox_expansion is
    set) or simply reused from the last detection (the default).
    """

    redetect_interval: int = 1
    fitness_checks: list[BBoxFitnessCheckConfig] = []
    keypoint_bbox_expansion: float | None = None
    keypoint_bbox_min_visibility: float = 0.0


class BBoxSmoothingConfig(BaseModel):
    """EMA parameters for bbox crop stabilisation."""

    alpha: float = 0.5


class KalmanKeypointSmoothingConfig(BaseModel):
    """Kalman filter (constant-velocity model) for keypoint temporal smoothing.

    Models each keypoint's x and y axes independently as [position, velocity]
    state vectors. The prediction step extrapolates using the estimated velocity,
    so occluded keypoints track the expected trajectory rather than freezing at
    the last position.

    process_noise_pos:  Q entry for position (pixels² per frame). Higher = trust
                        measurements more, less smoothing.
    process_noise_vel:  Q entry for velocity ((pixels/frame)² per frame). Higher =
                        velocity adapts faster to changes in motion.
    measurement_noise:  R (pixels²). Higher = trust measurements less, more smoothing.

    max_gap_frames and max_velocity work identically to KeypointSmoothingConfig.
    """

    kind: Literal["kalman"] = "kalman"
    process_noise_pos: float = 1.0
    process_noise_vel: float = 0.1
    measurement_noise: float = 10.0
    max_gap_frames: int | None = None
    max_velocity: float | None = None


class KeypointSmoothingConfig(BaseModel):
    """One-euro filter parameters for keypoint temporal smoothing.

    Gap filling: when a keypoint disappears for up to max_gap_frames consecutive
    frames, hold its last filtered position rather than returning NaN. Set to None
    to disable (current default — re-initialise on re-detection).

    Anomaly rejection: discard detections where a keypoint jumps more than
    max_velocity pixels per frame. Rejected points are treated like missing
    detections and are gap-filled if max_gap_frames is set. Set to None to
    disable (accept all detections).
    """

    kind: Literal["one_euro"] = "one_euro"
    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0
    max_gap_frames: int | None = None
    max_velocity: float | None = None
