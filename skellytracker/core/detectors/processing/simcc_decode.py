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


def get_simcc_maximum(
    simcc_x: NDArray,
    simcc_y: NDArray,
) -> tuple[NDArray, NDArray]:
    """Decode SIMCC heatmaps to (x, y) coordinates with confidence scores.

    Confidence is the *raw* SIMCC peak response averaged across the two axes,
    matching mmpose and rtmlib.  RTMPose is trained with a Gaussian-target KL
    loss, so a well-localised keypoint drives its winning bin to ~1.0 while an
    occluded one leaves a low, flat response — the peak height itself is the
    confidence signal.

    Do NOT softmax the bins first.  Softmax normalises each keypoint's response
    to sum to 1 over its bins, which discards peak height entirely: every
    keypoint then scores within a hair of the uniform floor (~1/Wx, ~1/Wy) and
    confident points stop separating from occluded ones.  Coordinates survive
    that mistake because argmax is monotone under softmax, so it goes unnoticed
    unless the confidences are inspected directly.

    Values are unbounded above (peaks above 1.0 are routine); the [0, 1] clip
    that ``Keypoints.visibility`` requires happens in
    ``rtmpose_letterbox_postprocess``, which is where skellytracker's contract
    starts.  This function stays byte-comparable with upstream mmpose/rtmlib.

    Parameters
    ----------
    simcc_x : np.ndarray  shape (N, K, Wx) — SIMCC responses for the x-axis.
    simcc_y : np.ndarray  shape (N, K, Wy) — SIMCC responses for the y-axis.

    Returns
    -------
    locs : np.ndarray  shape (N, K, 2)  x/y keypoint coordinates (bin indices).
    vals : np.ndarray  shape (N, K)     confidence = mean of the x and y peak
                                        responses.  Unbounded above; <= 0.0
                                        marks a non-detection (locs set to -1).
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)

    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    vals = 0.5 * (max_val_x + max_val_y)
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
    return keypoints.astype(np.float64), np.clip(scores, 0.0, 1.0).astype(np.float32)


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

    if len(raw) != 2:
        raise ValueError("SIMCC decoding requires exactly two output tensors")
    simcc_x, simcc_y = raw
    if (
        simcc_x.ndim != 3
        or simcc_y.ndim != 3
        or simcc_x.shape[0] != 1
        or simcc_y.shape[0] != 1
        or simcc_x.shape[1] != simcc_y.shape[1]
    ):
        raise ValueError("SIMCC decoding requires one image and matching landmark axes")
    expected_bins = np.asarray(model_input_size) * decode_spec.simcc_split_ratio
    if not np.array_equal(expected_bins, [simcc_x.shape[-1], simcc_y.shape[-1]]):
        raise ValueError(
            "SIMCC output bins disagree with declared input size and split ratio"
        )
    keypoints_xy, scores = decode_simcc(
        simcc_x=simcc_x,
        simcc_y=simcc_y,
        center=center,
        scale=scale,
        model_input_size=model_input_size,
        simcc_split_ratio=decode_spec.simcc_split_ratio,
    )
    return keypoints_xy[0], scores[0]
