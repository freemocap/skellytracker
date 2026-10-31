"""sfn26 numerical/public contracts that the upstream sidecar tests do not cover."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from skellytracker.core.detectors.keypoint_detectors._schema_loader import (
    load_point_names,
)
from skellytracker.core.detectors.keypoint_detectors.rtmpose.wholebody.rtmpose_wholebody_detector import (
    RTMPoseDetectorConfig,
    RTMPoseKeypointDetector,
    _SIDECAR,
    _SIDECAR_PATH,
)
from skellytracker.core.detectors.keypoint_detectors.rtmpose.rtmpose_preprocessing import (
    rtmpose_letterbox_preprocess,
)
from skellytracker.core.detectors.metadata import RTMPoseMetadata
from skellytracker.core.sessions.cpu_session import CpuSession


def test_native_points_are_mapped_to_existing_public_order_and_raw_confidence():
    names = load_point_names(_SIDECAR_PATH.with_name("rtmpose_wholebody.yaml"))
    count = len(names)
    simcc_x = np.zeros((1, count, 384), dtype=np.float32)
    simcc_y = np.zeros((1, count, 512), dtype=np.float32)
    indices = np.arange(count)
    peaks = np.linspace(0.1, 1.2, count, dtype=np.float32)
    simcc_x[0, indices, indices] = peaks
    simcc_y[0, indices, indices + 1] = peaks
    detector = RTMPoseKeypointDetector(RTMPoseDetectorConfig(), CpuSession())
    actual = detector.postprocess(
        [simcc_x, simcc_y],
        RTMPoseMetadata(center=np.array([96.0, 128.0]), scale=np.array([192.0, 256.0])),
    )
    permutation = np.asarray(
        [_SIDECAR.pose.tracked_points.index(name) for name in names]
    )
    expected = np.minimum(peaks[permutation], 1.0)
    missing = expected < 0.4
    expected[missing] = 0.0
    assert actual.names == names
    np.testing.assert_array_equal(actual.visibility, expected.astype(np.float64))
    np.testing.assert_array_equal(np.isnan(actual.xyz).all(axis=1), missing)
    np.testing.assert_array_equal(actual.xyz[~missing, 0], permutation[~missing] / 2.0)


def test_sidecar_preprocessing_matches_existing_rtmw_affine_and_normalization():
    image = np.random.default_rng(123).integers(0, 256, (121, 203, 3), dtype=np.uint8)
    detector = RTMPoseKeypointDetector(RTMPoseDetectorConfig(), CpuSession())
    with patch(
        "skellytracker.core.detectors.processing.image_preprocessing.build_normalization_fn",
        side_effect=AssertionError(
            "must prepare normalization before frame processing"
        ),
    ):
        actual, metadata = detector.preprocess(image)
    expected, center, scale = rtmpose_letterbox_preprocess(
        image,
        np.array([0.0, 0.0, 203.0, 121.0]),
        (192, 256),
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
    )
    np.testing.assert_array_equal(actual, expected.transpose(2, 0, 1))
    np.testing.assert_array_equal(metadata.center, center)
    np.testing.assert_array_equal(metadata.scale, scale)


def test_unknown_model_size_fails_instead_of_guessing_dimensions():
    with pytest.raises(KeyError):
        _ = RTMPoseDetectorConfig(model_name="undeclared-model").input_size
