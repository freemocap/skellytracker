# ruff: noqa: S101
import numpy as np
import pytest
from synthetic_metrics import evaluate_predictions


def test_pixel_conversion_visibility_and_low_confidence_failure_remain():
    truth = np.array([[[2.5, 3.5], [10.5, 3.5]]])
    predictions = np.array([[[2.0, 3.0, 0.9], [30.0, 3.0, 0.001]]])
    _, errors, _, report = evaluate_predictions(
        predictions,
        truth,
        np.ones((1, 2), bool),
        np.array([[True, False]]),
        ["one", "two"],
    )
    np.testing.assert_array_equal(errors, [[0, 20]])
    assert report["groups"]["visible"]["mean_error_px"] == 0
    assert report["groups"]["occluded"]["mean_error_px"] == 20
    assert report["groups"]["all_valid"]["pck"]["10"] == 0.5
    assert report["groups"]["all_valid"]["euclidean_rmse_px"] == pytest.approx(np.sqrt(200))
    assert report["confidence_diagnostic"]["included_count"] == 1
    assert report["confidence_diagnostic"]["excluded_valid_count"] == 1
    assert report["confidence_diagnostic"]["coverage"] == .5


def test_nonfinite_prediction_counts_as_failure_and_invalid_geometry_excluded():
    truth = np.ones((1, 3, 2))
    predictions = np.array([[[np.nan, 0, 0.9], [0.5, 0.5, np.nan], [1000, 1000, 0.9]]])
    _, _, _, report = evaluate_predictions(
        predictions,
        truth,
        np.array([[True, True, False]]),
        np.ones((1, 3), bool),
        ["one", "two", "three"],
    )
    group = report["groups"]["all_valid"]
    assert group["target_count"] == 2
    assert group["nonfinite_prediction_count"] == 2
    assert group["pck"]["20"] == 0
    assert report["projection_invalid_count"] == 1


def test_per_feature_metrics_keep_native_identity_order():
    truth = np.ones((2, 2, 2)) * 0.5
    values = np.array([[[1.0, 0, 0.8], [2.0, 0, 0.8]], [[3.0, 0, 0.8], [4.0, 0, 0.8]]])
    _, _, _, report = evaluate_predictions(
        values, truth, np.ones((2, 2), bool), np.ones((2, 2), bool), ["second", "first"]
    )
    assert [entry["feature_name"] for entry in report["per_feature"]] == [
        "second",
        "first",
    ]
    assert [entry["all_valid"]["mean_error_px"] for entry in report["per_feature"]] == [
        2,
        3,
    ]


def test_mismatched_tensor_axes_fail():
    with pytest.raises(ValueError, match="matching geometry"):
        evaluate_predictions(
            np.zeros((2, 3, 3)),
            np.zeros((2, 4, 2)),
            np.ones((2, 4), bool),
            np.ones((2, 4), bool),
            list("abcd"),
        )
