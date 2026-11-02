import numpy as np
import pytest
from reference_contract import validate_predictions


def valid_fixture(keypoint_count=7):
    values = np.zeros((5, 1, keypoint_count, 3), dtype=np.float32)
    metadata = {
        "image_paths": [f"image_{index}.png" for index in range(5)],
        "keypoint_names": [f"feature_{index}" for index in range(keypoint_count)],
        "component_names": ["x", "y", "confidence"],
        "coordinate_space": "original image pixels",
        "shape": list(values.shape),
    }
    return values, metadata


@pytest.mark.parametrize("keypoint_count", [1, 4, 7, 19])
def test_keypoint_count_comes_from_metadata(keypoint_count):
    validate_predictions(*valid_fixture(keypoint_count))


def test_reordered_components_are_not_silently_accepted():
    values, metadata = valid_fixture()
    metadata["component_names"] = ["confidence", "x", "y"]
    with pytest.raises(ValueError, match="requires x, y, confidence"):
        validate_predictions(values, metadata)


def test_duplicate_identities_are_rejected():
    values, metadata = valid_fixture()
    metadata["keypoint_names"][1] = metadata["keypoint_names"][0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_predictions(values, metadata)


@pytest.mark.parametrize("bad_confidence", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_reference_predictions_are_rejected(bad_confidence):
    values, metadata = valid_fixture()
    values[0, 0, 0, 2] = bad_confidence
    with pytest.raises(ValueError):
        validate_predictions(values, metadata)


def test_axis_mismatch_is_rejected():
    values, metadata = valid_fixture()
    with pytest.raises(ValueError, match="axes"):
        validate_predictions(values[:, :, :-1], metadata)
