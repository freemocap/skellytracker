"""Use cached test weights without downloading models during pytest collection."""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from skellytracker.core.sessions.model_registry import DEFAULT_CACHE_DIR, ModelSource
from skellytracker.core.sessions.onnx_model_spec import OnnxModelSpec


def cached_model_spec(spec: OnnxModelSpec) -> OnnxModelSpec:
    roots = [DEFAULT_CACHE_DIR]
    extra_cache = os.environ.get("SKELLYTRACKER_TEST_MODEL_CACHE")
    if extra_cache:
        roots.insert(0, Path(extra_cache))
    for root in roots:
        path = root / spec.expected_filename
        if path.is_file():
            return replace(spec, source=ModelSource(local_path=str(path)))
    pytest.skip(f"Optional model weights are not cached: {spec.name}")


def assert_original_pose_processing(detector, image, native_names):
    """Compare against the existing preprocessing/decoding on the same ORT session.

    Upstream golden files omit provider and precision and some encode the older
    softmax confidence threshold. This comparison isolates the sidecar migration
    while keeping inference configuration and the raw-score contract explicit.
    """
    import numpy as np
    from skellytracker.core.detectors.keypoint_detectors.rtmpose.rtmpose_preprocessing import (
        rtmpose_letterbox_preprocess,
        rtmpose_letterbox_postprocess,
    )

    height, width = image.shape[:2]
    input_height, input_width = detector.config.input_size
    original, center, scale = rtmpose_letterbox_preprocess(
        image,
        np.array([0.0, 0.0, float(width), float(height)]),
        (input_width, input_height),
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
    )
    tensor = np.ascontiguousarray(original.transpose(2, 0, 1))
    actual_tensor, metadata = detector.preprocess(image)
    np.testing.assert_array_equal(actual_tensor, tensor)
    np.testing.assert_array_equal(metadata.center, center)
    np.testing.assert_array_equal(metadata.scale, scale)
    input_name = (
        detector.session.get_session(detector.config.model_name).get_inputs()[0].name
    )
    raw = detector.session.run(
        detector.config.model_name, {input_name: tensor[np.newaxis]}
    )
    coordinates, scores = rtmpose_letterbox_postprocess(
        *raw, center, scale, (input_width, input_height)
    )
    actual = detector.detect(image)
    assert len(actual.names) == len(native_names)
    permutation = [native_names.index(name) for name in actual.names]
    expected_scores = scores[0, permutation].astype(np.float64)
    expected_xyz = np.zeros((len(native_names), 3), dtype=np.float64)
    expected_xyz[:, :2] = coordinates[0, permutation]
    missing = expected_scores < detector.config.confidence_threshold
    expected_scores[missing] = 0.0
    expected_xyz[missing] = np.nan
    np.testing.assert_allclose(
        actual.xyz, expected_xyz, rtol=0, atol=1e-5, equal_nan=True
    )
    np.testing.assert_array_equal(actual.visibility, expected_scores)
