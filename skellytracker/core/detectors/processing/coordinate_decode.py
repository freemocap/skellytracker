"""Decode declared x/y/confidence tensors into owned keypoint value arrays."""

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.detectors.metadata import ImageTransform


def decode_coordinates(
    values: NDArray[np.float32],
    transform: ImageTransform,
    *,
    keypoint_count: int,
    confidence_threshold: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Invert preprocessing once; leave the outer stage's crop translation to it.

    Each invalid prediction becomes all-NaN coordinates and zero confidence.
    The returned arrays own their data and never alias the inference output.
    """
    if (
        values.dtype != np.float32
        or values.ndim != 3
        or values.shape[1:] != (keypoint_count, 3)
    ):
        raise ValueError(
            "Coordinate output must be float32 [batch, declared keypoints, x/y/confidence]"
        )
    if keypoint_count < 1 or not 0 <= confidence_threshold <= 1:
        raise ValueError(
            "Require positive keypoint count and confidence threshold in [0, 1]"
        )
    transform_values = np.array(
        (transform.scale_x, transform.scale_y, transform.offset_x, transform.offset_y)
    )
    if not np.isfinite(transform_values).all() or np.any(transform_values[:2] <= 0):
        raise ValueError(
            "Coordinate transform requires finite offsets and positive finite scales"
        )
    scores = values[..., 2].astype(np.float64, copy=True)
    if np.any(np.isfinite(scores) & ((scores < 0) | (scores > 1))):
        raise ValueError("Coordinate confidence must be in [0, 1]")
    valid = np.isfinite(values).all(axis=-1) & (scores >= confidence_threshold)
    xyz = np.zeros(values.shape, dtype=np.float64)
    xyz[..., :2] = transform.restore_points(values[..., :2])
    xyz[~valid] = np.nan
    scores[~valid] = 0.0
    return xyz, scores
