"""Generic SIMCC pose decode: argmax + affine-unprojection, name-indexed.

SIMCC ("SimCC") is a pose-decode head shared by multiple model families in
the mmpose ecosystem (RTMPose, RTMW, and others) — nothing here is specific
to any one of them. `simcc_pose_decode` is the sidecar-driven entry point
detectors call from `postprocess()`; `get_simcc_maximum`/`decode_simcc` are
the underlying math (vendored from rtmlib's
``tools/pose_estimation/rtmpose.py``), usable directly by anything that
needs raw SIMCC decode without going through a sidecar.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.sidecar.model import PoseDecodeSpec


def _stable_softmax(x: NDArray) -> NDArray:
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def get_simcc_maximum(
    simcc_x: NDArray,
    simcc_y: NDArray,
) -> tuple[NDArray, NDArray]:
    """Decode SIMCC heatmaps to (x, y) coordinates with confidence scores.

    Applies softmax to raw logits, then returns the peak probability as
    confidence.  Value is in [0, 1]: higher = more mass concentrated in
    the winning bin.  A uniform distribution over N bins gives ~1/N.

    Parameters
    ----------
    simcc_x : np.ndarray  shape (N, K, Wx) — raw SIMCC logits for x-axis.
    simcc_y : np.ndarray  shape (N, K, Wy) — raw SIMCC logits for y-axis.

    Returns
    -------
    locs : np.ndarray  shape (N, K, 2)  x/y keypoint coordinates.
    vals : np.ndarray  shape (N, K)     confidence = average peak softmax
                                        probability across x and y axes.
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    px = _stable_softmax(simcc_x)
    py = _stable_softmax(simcc_y)

    x_locs = np.argmax(px, axis=1)
    y_locs = np.argmax(py, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)

    max_px = np.amax(px, axis=1)
    max_py = np.amax(py, axis=1)

    vals = 0.5 * (max_px + max_py)
    locs[vals <= 0.0] = -1

    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


def decode_simcc(
    simcc_x: NDArray,
    simcc_y: NDArray,
    center: NDArray,
    scale: NDArray,
    model_input_size: tuple[int, int],
    simcc_split_ratio: float = 2.0,
) -> tuple[NDArray, NDArray]:
    """Decode SIMCC outputs back to original image coordinates.

    Parameters
    ----------
    simcc_x, simcc_y : (1, K, W) / (1, K, H) SIMCC heatmaps.
    center : (2,) bbox center in image coords (from affine_crop.bbox_xyxy2cs).
    scale : (2,) bbox scale.
    model_input_size : (H, W) model input.
    simcc_split_ratio : label resolution divisor.

    Returns
    -------
    keypoints : (1, K, 2) float64  image-coordinate keypoints.
    scores : (1, K) float32
    """
    locs, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints = locs / simcc_split_ratio
    keypoints = keypoints / np.asarray(model_input_size) * scale
    keypoints = keypoints + center - scale / 2
    return keypoints.astype(np.float64), scores.astype(np.float32)


_SIMCC_SPLIT_RATIO: float = 2.0  # unchanged from the pre-migration hardcoded constant


def simcc_pose_decode(
    raw: Any,
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
    model_input_size: tuple[int, int],
    decode_spec: PoseDecodeSpec,
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    """Decode raw SIMCC outputs for one image into (xy, scores), dispatching
    on a sidecar's `PoseDecodeSpec`.

    `raw` is `(simcc_x, simcc_y)`, each shape `(1, K, bins)`.
    `model_input_size` is `(W, H)`.

    Returns `(kpts_xy, kpt_scores)` — shapes `(K, 2)` and `(K,)` — indexed in
    `pose.tracked_points` order (native model output order); the caller
    zips these directly against `tracked_points`, no permutation.
    """
    if decode_spec.method != "simcc":
        raise NotImplementedError(
            f"pose.decode.method={decode_spec.method!r} is not yet implemented "
            "(only 'simcc' is)"
        )
    if decode_spec.is_3d:
        raise NotImplementedError("pose.decode.is_3d is not yet implemented")

    simcc_x, simcc_y = raw
    keypoints_xy, scores = decode_simcc(
        simcc_x=simcc_x,
        simcc_y=simcc_y,
        center=center,
        scale=scale,
        model_input_size=model_input_size,
        simcc_split_ratio=_SIMCC_SPLIT_RATIO,
    )
    return keypoints_xy[0], scores[0]
