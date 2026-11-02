import numpy as np
import pytest

from skellytracker.core.detectors.metadata import ImageTransform
from skellytracker.core.detectors.processing.coordinate_decode import decode_coordinates


@pytest.mark.parametrize("count", [1, 4, 17])
def test_coordinates_owned_native_order_and_inverse_transform(count):
    values = np.zeros((3, count, 3), dtype=np.float32)
    values[..., 0] = np.arange(count) + 10
    values[..., 1] = 20
    values[..., 2] = 0.5
    xyz, scores = decode_coordinates(
        values,
        ImageTransform(2.0, 4.0, 6.0, 8.0),
        keypoint_count=count,
        confidence_threshold=0.5,
    )
    np.testing.assert_array_equal(
        xyz[..., 0], np.broadcast_to((np.arange(count) + 4) / 2, (3, count))
    )
    np.testing.assert_array_equal(xyz[..., 1], 3)
    np.testing.assert_array_equal(xyz[..., 2], 0)
    values[:] = 0
    np.testing.assert_array_equal(scores, 0.5)
    assert xyz.flags.owndata and scores.flags.owndata


def test_missing_predictions_are_explicit():
    values = np.array(
        [[[1, 2, 0.2], [np.nan, 2, 0.9], [1, 2, np.nan], [1, 2, 0.5]]], dtype=np.float32
    )
    xyz, scores = decode_coordinates(
        values, ImageTransform(1.0, 1.0), keypoint_count=4, confidence_threshold=0.5
    )
    assert np.isnan(xyz[0, :3]).all()
    np.testing.assert_array_equal(scores[0, :3], 0)
    assert np.isfinite(xyz[0, 3]).all()


def test_explicit_zero_threshold_preserves_finite_zero_score():
    values = np.array([[[1, 2, 0]]], dtype=np.float32)
    xyz, scores = decode_coordinates(
        values, ImageTransform(1.0, 1.0), keypoint_count=1, confidence_threshold=0.0
    )
    np.testing.assert_array_equal(xyz, [[[1, 2, 0]]])
    np.testing.assert_array_equal(scores, [[0]])


def test_malformed_confidence_and_transform_fail():
    values = np.array([[[1, 2, 1.1]]], dtype=np.float32)
    with pytest.raises(ValueError, match="confidence"):
        decode_coordinates(
            values, ImageTransform(1.0, 1.0), keypoint_count=1, confidence_threshold=0.5
        )
    with pytest.raises(ValueError, match="transform"):
        decode_coordinates(
            values, ImageTransform(0.0, 1.0), keypoint_count=1, confidence_threshold=0.5
        )
