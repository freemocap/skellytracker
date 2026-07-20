from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.temporal_processing.temporal_processing_config import (
    KalmanKeypointSmoothingConfig,
    KeypointSmoothingConfig,
)
from skellytracker.core.tracker.tracker_state import KeypointSmoothingState


# ---------------------------------------------------------------------------
# One-euro filter
# ---------------------------------------------------------------------------

def apply_one_euro_filter(
    raw_keypoints: Keypoints,
    state: KeypointSmoothingState,
    dt: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
    max_gap_frames: int | None = None,
    max_velocity: float | None = None,
) -> tuple[Keypoints, KeypointSmoothingState]:
    """Apply a one-euro filter with optional gap filling and anomaly rejection.

    One-euro filter: adapts cutoff frequency based on motion speed — low speed
    gives more smoothing; high speed gives less (avoids lag).

    Anomaly rejection: if max_velocity is set, detections where a keypoint jumps
    more than max_velocity pixels/frame are discarded and treated as missing for
    that frame.

    Gap filling: if max_gap_frames is set, keypoints that have been missing (or
    rejected) for <= max_gap_frames consecutive frames hold their last filtered
    position rather than returning NaN. State is frozen during the gap so the
    filter resumes cleanly when detection returns.

    dt is elapsed time since the last frame (seconds when timestamp_ms is
    available, else 1.0 frame units).
    """
    x = raw_keypoints.xyz[:, 0].copy()
    y = raw_keypoints.xyz[:, 1].copy()
    raw_valid: NDArray[np.bool_] = ~np.isnan(x) & ~np.isnan(y)
    n = len(x)

    if state.x_prev is None:
        new_state = KeypointSmoothingState(
            x_prev=np.where(raw_valid, x, np.nan),
            y_prev=np.where(raw_valid, y, np.nan),
            dx_prev=np.zeros(n),
            dy_prev=np.zeros(n),
            frames_since_detection=np.where(raw_valid, 0, 1).astype(np.int32),
        )
        return raw_keypoints, new_state

    x_prev = state.x_prev
    y_prev = state.y_prev
    dx_prev = state.dx_prev if state.dx_prev is not None else np.zeros(n)
    dy_prev = state.dy_prev if state.dy_prev is not None else np.zeros(n)
    frames_since = (
        state.frames_since_detection
        if state.frames_since_detection is not None
        else np.zeros(n, dtype=np.int32)
    )
    prev_valid: NDArray[np.bool_] = ~np.isnan(x_prev)

    # 1. Anomaly rejection
    detected = raw_valid.copy()
    if max_velocity is not None:
        dist = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)
        anomalous = raw_valid & prev_valid & (dist > max_velocity)
        detected = raw_valid & ~anomalous

    # 2. Per-point gap counter
    new_frames_since: NDArray[np.int32] = np.where(detected, 0, frames_since + 1).astype(np.int32)

    # 3. Gap filling
    if max_gap_frames is not None:
        gap_fill: NDArray[np.bool_] = (
            ~detected & (new_frames_since <= max_gap_frames) & prev_valid
        )
    else:
        gap_fill = np.zeros(n, dtype=bool)

    # 4. One-euro filter
    filter_valid = detected & prev_valid
    init_valid = detected & ~prev_valid

    alpha_d = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * d_cutoff * dt))
    dx = np.where(filter_valid, alpha_d * (x - x_prev) / dt + (1.0 - alpha_d) * dx_prev, dx_prev)
    dy = np.where(filter_valid, alpha_d * (y - y_prev) / dt + (1.0 - alpha_d) * dy_prev, dy_prev)

    alphas_x = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * (min_cutoff + beta * np.abs(dx)) * dt))
    alphas_y = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * (min_cutoff + beta * np.abs(dy)) * dt))

    # 5. Output:  filtered | initialized | gap-held | NaN
    x_hat = np.where(filter_valid, alphas_x * x + (1.0 - alphas_x) * x_prev,
             np.where(init_valid, x,
             np.where(gap_fill, x_prev, np.nan)))
    y_hat = np.where(filter_valid, alphas_y * y + (1.0 - alphas_y) * y_prev,
             np.where(init_valid, y,
             np.where(gap_fill, y_prev, np.nan)))

    xyz_out = raw_keypoints.xyz.copy()
    xyz_out[:, 0] = x_hat
    xyz_out[:, 1] = y_hat
    filtered = Keypoints(names=raw_keypoints.names, xyz=xyz_out, visibility=raw_keypoints.visibility.copy())

    # 6. State update — freeze during gap; reset to NaN when track is lost
    new_x_prev = np.where(detected, x_hat, np.where(gap_fill, x_prev, np.nan))
    new_y_prev = np.where(detected, y_hat, np.where(gap_fill, y_prev, np.nan))
    new_dx = np.where(detected, dx, dx_prev)
    new_dy = np.where(detected, dy, dy_prev)

    new_state = KeypointSmoothingState(
        x_prev=new_x_prev, y_prev=new_y_prev,
        dx_prev=new_dx, dy_prev=new_dy,
        frames_since_detection=new_frames_since,
    )
    return filtered, new_state


# ---------------------------------------------------------------------------
# Kalman filter (constant-velocity model)
# ---------------------------------------------------------------------------

def apply_kalman_filter(
    raw_keypoints: Keypoints,
    state: KeypointSmoothingState,
    dt: float,
    process_noise_pos: float,
    process_noise_vel: float,
    measurement_noise: float,
    max_gap_frames: int | None = None,
    max_velocity: float | None = None,
) -> tuple[Keypoints, KeypointSmoothingState]:
    """Apply a constant-velocity Kalman filter to all keypoints.

    Each axis (x, y) is modelled independently as a 2-state system
    [position, velocity].  The prediction step extrapolates using the
    estimated velocity, so occluded keypoints track the expected trajectory
    rather than holding the last seen position.

    max_gap_frames and max_velocity work the same as in apply_one_euro_filter.
    During a gap the Kalman prediction is used as output (position + velocity *
    dt), which is more accurate than holding the last position for moving subjects.
    """
    x_meas = raw_keypoints.xyz[:, 0].copy()
    y_meas = raw_keypoints.xyz[:, 1].copy()
    raw_valid: NDArray[np.bool_] = ~np.isnan(x_meas) & ~np.isnan(y_meas)
    n = len(x_meas)

    # Constant matrices
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.array([[process_noise_pos, 0.0], [0.0, process_noise_vel]])
    P_init = np.diag([measurement_noise * 10.0, process_noise_vel * 100.0])

    # -----------------------------------------------------------------------
    # First frame
    # -----------------------------------------------------------------------
    if state.x_prev is None:
        P_all = np.tile(P_init, (n, 1, 1))
        new_state = KeypointSmoothingState(
            x_prev=np.where(raw_valid, x_meas, np.nan),
            y_prev=np.where(raw_valid, y_meas, np.nan),
            vx_est=np.zeros(n),
            vy_est=np.zeros(n),
            Px=P_all.copy(),
            Py=P_all.copy(),
            frames_since_detection=np.where(raw_valid, 0, 1).astype(np.int32),
        )
        return raw_keypoints, new_state

    x_est = state.x_prev
    y_est = state.y_prev
    vx_est = state.vx_est if state.vx_est is not None else np.zeros(n)
    vy_est = state.vy_est if state.vy_est is not None else np.zeros(n)
    Px = state.Px if state.Px is not None else np.tile(P_init, (n, 1, 1))
    Py = state.Py if state.Py is not None else np.tile(P_init, (n, 1, 1))
    frames_since = (
        state.frames_since_detection
        if state.frames_since_detection is not None
        else np.zeros(n, dtype=np.int32)
    )
    prev_valid: NDArray[np.bool_] = ~np.isnan(x_est)

    # -----------------------------------------------------------------------
    # Anomaly rejection
    # -----------------------------------------------------------------------
    detected = raw_valid.copy()
    if max_velocity is not None:
        dist = np.sqrt((x_meas - x_est) ** 2 + (y_meas - y_est) ** 2)
        anomalous = raw_valid & prev_valid & (dist > max_velocity)
        detected = raw_valid & ~anomalous

    new_frames_since: NDArray[np.int32] = np.where(detected, 0, frames_since + 1).astype(np.int32)

    if max_gap_frames is not None:
        gap_fill: NDArray[np.bool_] = (
            ~detected & (new_frames_since <= max_gap_frames) & prev_valid
        )
    else:
        gap_fill = np.zeros(n, dtype=bool)

    # -----------------------------------------------------------------------
    # Predict step (for all points that have a previous estimate)
    # -----------------------------------------------------------------------
    x_pred = np.where(prev_valid, x_est + vx_est * dt, np.nan)
    y_pred = np.where(prev_valid, y_est + vy_est * dt, np.nan)
    vx_pred = vx_est.copy()
    vy_pred = vy_est.copy()

    # Px_pred = F @ Px @ F.T + Q  (vectorised over N)
    FPx = np.einsum("ij,njk->nik", F, Px)
    Px_pred: NDArray[np.float64] = np.einsum("nij,kj->nik", FPx, F) + Q
    FPy = np.einsum("ij,njk->nik", F, Py)
    Py_pred: NDArray[np.float64] = np.einsum("nij,kj->nik", FPy, F) + Q

    # -----------------------------------------------------------------------
    # Update step for detected points that have a prior estimate
    # -----------------------------------------------------------------------
    update_valid = detected & prev_valid
    init_valid = detected & ~prev_valid  # first detection for this point

    # Innovation covariance: S = P_pred[0,0] + R
    Sx = np.where(update_valid, Px_pred[:, 0, 0] + measurement_noise, 1.0)
    Sy = np.where(update_valid, Py_pred[:, 0, 0] + measurement_noise, 1.0)

    # Kalman gain: K = P_pred[:, :, 0] / S  — shape (N, 2)
    Kx: NDArray[np.float64] = Px_pred[:, :, 0] / Sx[:, None]
    Ky: NDArray[np.float64] = Py_pred[:, :, 0] / Sy[:, None]

    innov_x = np.where(update_valid, x_meas - x_pred, 0.0)
    innov_y = np.where(update_valid, y_meas - y_pred, 0.0)

    x_updated = x_pred + Kx[:, 0] * innov_x
    y_updated = y_pred + Ky[:, 0] * innov_y
    vx_updated = vx_pred + Kx[:, 1] * innov_x
    vy_updated = vy_pred + Ky[:, 1] * innov_y

    # Covariance update: P_new = (I - K @ H) @ P_pred  where H = [1, 0]
    # (I - K @ H) = [[1-K0, 0], [-K1, 1]]
    Kx0, Kx1 = Kx[:, 0], Kx[:, 1]
    Px_new = np.stack([
        (1.0 - Kx0)[:, None] * Px_pred[:, 0, :],
        -Kx1[:, None] * Px_pred[:, 0, :] + Px_pred[:, 1, :],
    ], axis=1)
    Ky0, Ky1 = Ky[:, 0], Ky[:, 1]
    Py_new = np.stack([
        (1.0 - Ky0)[:, None] * Py_pred[:, 0, :],
        -Ky1[:, None] * Py_pred[:, 0, :] + Py_pred[:, 1, :],
    ], axis=1)

    # Select covariance: updated for update_valid, predicted for gap/predict-only, P_init otherwise
    Px_out: NDArray[np.float64] = np.where(
        update_valid[:, None, None], Px_new,
        np.where(prev_valid[:, None, None], Px_pred, P_init),
    )
    Py_out: NDArray[np.float64] = np.where(
        update_valid[:, None, None], Py_new,
        np.where(prev_valid[:, None, None], Py_pred, P_init),
    )

    # -----------------------------------------------------------------------
    # Output coordinates
    #   update_valid  → Kalman-updated position
    #   init_valid    → raw measurement (first detection)
    #   gap_fill      → Kalman-predicted position (velocity extrapolation)
    #   else          → NaN
    # -----------------------------------------------------------------------
    x_out = np.where(update_valid, x_updated,
             np.where(init_valid, x_meas,
             np.where(gap_fill, x_pred, np.nan)))
    y_out = np.where(update_valid, y_updated,
             np.where(init_valid, y_meas,
             np.where(gap_fill, y_pred, np.nan)))

    xyz_out = raw_keypoints.xyz.copy()
    xyz_out[:, 0] = x_out
    xyz_out[:, 1] = y_out
    filtered = Keypoints(names=raw_keypoints.names, xyz=xyz_out, visibility=raw_keypoints.visibility.copy())

    # -----------------------------------------------------------------------
    # State update
    # -----------------------------------------------------------------------
    new_x_est = np.where(update_valid, x_updated,
                np.where(init_valid, x_meas,
                np.where(gap_fill, x_pred, np.nan)))
    new_y_est = np.where(update_valid, y_updated,
                np.where(init_valid, y_meas,
                np.where(gap_fill, y_pred, np.nan)))
    new_vx = np.where(update_valid, vx_updated,
              np.where(init_valid | gap_fill, vx_pred, 0.0))
    new_vy = np.where(update_valid, vy_updated,
              np.where(init_valid | gap_fill, vy_pred, 0.0))

    new_state = KeypointSmoothingState(
        x_prev=new_x_est, y_prev=new_y_est,
        vx_est=new_vx, vy_est=new_vy,
        Px=Px_out, Py=Py_out,
        frames_since_detection=new_frames_since,
    )
    return filtered, new_state


# ---------------------------------------------------------------------------
# Smoother classes and factory
# ---------------------------------------------------------------------------

@dataclass
class OneEuroFilter:
    min_cutoff: float
    beta: float
    d_cutoff: float
    max_gap_frames: int | None = None
    max_velocity: float | None = None

    def smooth(
        self,
        keypoints: Keypoints,
        state: KeypointSmoothingState,
        dt: float,
    ) -> tuple[Keypoints, KeypointSmoothingState]:
        return apply_one_euro_filter(
            keypoints, state, dt,
            self.min_cutoff, self.beta, self.d_cutoff,
            self.max_gap_frames, self.max_velocity,
        )

    @classmethod
    def from_config(cls, config: KeypointSmoothingConfig) -> OneEuroFilter:
        return cls(
            min_cutoff=config.min_cutoff,
            beta=config.beta,
            d_cutoff=config.d_cutoff,
            max_gap_frames=config.max_gap_frames,
            max_velocity=config.max_velocity,
        )


@dataclass
class KalmanFilter:
    process_noise_pos: float
    process_noise_vel: float
    measurement_noise: float
    max_gap_frames: int | None = None
    max_velocity: float | None = None

    def smooth(
        self,
        keypoints: Keypoints,
        state: KeypointSmoothingState,
        dt: float,
    ) -> tuple[Keypoints, KeypointSmoothingState]:
        return apply_kalman_filter(
            keypoints, state, dt,
            self.process_noise_pos, self.process_noise_vel, self.measurement_noise,
            self.max_gap_frames, self.max_velocity,
        )

    @classmethod
    def from_config(cls, config: KalmanKeypointSmoothingConfig) -> KalmanFilter:
        return cls(
            process_noise_pos=config.process_noise_pos,
            process_noise_vel=config.process_noise_vel,
            measurement_noise=config.measurement_noise,
            max_gap_frames=config.max_gap_frames,
            max_velocity=config.max_velocity,
        )


def make_keypoint_filter(
    config: KeypointSmoothingConfig | KalmanKeypointSmoothingConfig,
) -> OneEuroFilter | KalmanFilter:
    if isinstance(config, KalmanKeypointSmoothingConfig):
        return KalmanFilter.from_config(config)
    return OneEuroFilter.from_config(config)
