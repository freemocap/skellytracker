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
    # Floor on how much the keypoint-predicted crop may shrink from one frame
    # to the next (as a fraction of the previous crop's width/height). This is
    # a *rate limit*, not a one-time expansion: relative expansion alone can't
    # stop collapse, because expanding a percentage of an already-shrunk tight
    # box still yields a smaller box than before whenever fewer keypoints are
    # visible than last frame — a self-reinforcing shrink loop with no stable
    # equilibrium. Clamping the shrink rate breaks that loop directly. Set to
    # None to disable (unbounded shrink, old behavior).
    #
    # Kept close to 1.0 so a keypoint sitting near the crop edge (e.g. a
    # shoulder at the frame boundary) survives many consecutive misses before
    # the crop drifts past its location — once that happens the keypoint
    # detector never sees that region again until the next redetect, so the
    # per-frame rate needs to stay slow relative to redetect_interval, not
    # just nonzero.
    min_shrink_ratio_per_frame: float | None = 0.999
    # Floor on predicted crop width/height as a fraction of the object
    # detector's most recent actual box (not the EMA-smoothed crop). The
    # per-frame rate limit above only paces the shrink; over a long
    # redetect_interval it still eventually converges on the tight
    # keypoint-only box. This ties the crop back to what the detector — which
    # sees the whole frame and isn't blind to points outside the current crop
    # — actually measured, so a long run of partial keypoint visibility can't
    # shrink the crop indefinitely. Set to None to disable.
    min_detected_bbox_ratio: float | None = 0.5
    # Hard floor on predicted crop width/height in pixels, regardless of the
    # shrink-rate clamp above. Purely a last-resort guard against a
    # zero-or-negative-size crop reaching cv2.warpAffine.
    min_bbox_size_px: float = 80.0


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


class KeypointResetPolicyConfig(BaseModel):
    """Resets a keypoint detector's internal temporal state after consecutive misses.

    Some keypoint detectors (e.g. MediaPipe's VIDEO-mode PoseLandmarker) maintain
    an internal track-then-detect pipeline: after tracking is lost, the detector
    can get stuck silently returning empty results even when the subject is
    clearly visible, because re-detection is not automatically retried. Calling
    detector.reset_temporal_state() discards the stale state and forces a fresh
    full detection on the next frame.

    A "miss" is a frame where the detector returns zero valid keypoints, checked
    before any confidence/visibility filtering — a detector that found the person
    but with low-visibility limbs does not count as a miss.

    Set max_consecutive_misses to None to disable (the default — no behavior
    change for detectors that don't need this, e.g. stateless ONNX detectors).
    """

    max_consecutive_misses: int | None = None


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
