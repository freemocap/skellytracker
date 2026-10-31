import cv2
import numpy as np
import pytest

from skellytracker.core.detectors.metadata import ImageTransform
from skellytracker.core.detectors.processing.image_preprocessing import (
    preprocess_image,
    fit_preprocess,
)
from skellytracker.core.detectors.processing.object_detection_decode import (
    decode_nms_baked_in,
    decode_prenms,
)
from skellytracker.core.sidecar.model import InputSpec, DetectionDecodeSpec


def input_spec(**resize):
    return InputSpec(
        name="images", dtype={"fp32": "float32"}, normalization="none", resize=resize
    )


@pytest.mark.parametrize("rounding,height", [("half_even", 2), ("half_up", 3)])
def test_letterbox_applies_rounding_centering_and_exact_inverse(rounding, height):
    source = np.full((5, 10, 3), 200, dtype=np.uint8)
    tensor, transform = preprocess_image(
        source,
        (5, 5),
        input_spec(
            method="letterbox",
            pad_position="center",
            rounding=rounding,
            pad_value=[10, 20, 30],
            interpolation="nearest",
        ),
    )
    expected = np.full((5, 5, 3), [10, 20, 30], dtype=np.uint8)
    expected[1 : 1 + height] = 200
    np.testing.assert_array_equal(tensor, expected.transpose(2, 0, 1))
    assert isinstance(transform, ImageTransform)
    np.testing.assert_allclose(
        transform.restore_points(np.array([[0.0, 1.0], [5.0, 1.0 + height]])),
        [[0.0, 0.0], [10.0, 5.0]],
    )


def test_stretch_records_independent_axis_scales():
    source = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    tensor, transform = preprocess_image(source, (12, 16), input_spec(method="stretch"))
    np.testing.assert_array_equal(
        tensor, cv2.resize(source, (16, 12)).transpose(2, 0, 1)
    )
    assert transform.scale_x == 2.0
    assert transform.scale_y == 3.0
    np.testing.assert_array_equal(
        transform.restore_boxes(np.array([[2.0, 3.0, 14.0, 9.0]])),
        [[1.0, 1.0, 7.0, 3.0]],
    )


@pytest.mark.parametrize(
    "pad_value,expected", [(114, [114, 114, 114]), ([10, 20, 30], [10, 20, 30])]
)
def test_fit_padding_fills_all_channels_and_fractional_offsets_roundtrip(
    pad_value, expected
):
    source = np.zeros((3, 4, 3), dtype=np.uint8)
    tensor, transform = preprocess_image(
        source, (8, 6), input_spec(method="fit", pad_value=pad_value)
    )
    np.testing.assert_array_equal(tensor[:, 0, 0], expected)
    assert transform.offset_y == 1.75
    np.testing.assert_allclose(
        transform.restore_points(np.array([[0.0, 1.75], [6.0, 6.25]])),
        [[0.0, 0.0], [4.0, 3.0]],
    )
    _, (scale, left, top) = fit_preprocess(source, (8, 6), pad_value=pad_value)
    assert (scale, left, top) == (1.5, 0.0, 1.75)


@pytest.mark.parametrize("method", ["fit", "stretch", "letterbox"])
def test_detection_decoders_apply_resize_inverse_without_mutating_raw_arrays(method):
    source = np.zeros((3, 7, 3), dtype=np.uint8)
    resize = dict(method=method)
    if method == "letterbox":
        resize.update(pad_position="center", rounding="half_up")
    _, transform = preprocess_image(source, (10, 12), input_spec(**resize))
    source_boxes = np.array([[1.0, 0.5, 6.0, 2.5]])
    model_boxes = source_boxes.reshape(-1, 2).copy()
    model_boxes *= (transform.scale_x, transform.scale_y)
    model_boxes += (transform.offset_x, transform.offset_y)
    model_boxes = model_boxes.reshape(-1, 4)
    original = model_boxes.copy()
    spec = DetectionDecodeSpec()
    actual, _ = decode_prenms(model_boxes, np.array([[0.9]]), transform, 0.5, 0.5, spec)
    np.testing.assert_allclose(actual, source_boxes, rtol=0, atol=1e-12)
    baked = np.concatenate((model_boxes, np.array([[0.9]])), axis=1)[None]
    actual, _ = decode_nms_baked_in(baked, transform, 0.5, spec)
    np.testing.assert_allclose(actual, source_boxes, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(model_boxes, original)


def test_unspecified_letterbox_preserves_existing_scalar_ratio():
    _, transform = preprocess_image(
        np.zeros((3, 7, 3), dtype=np.uint8), (10, 12), input_spec(method="letterbox")
    )
    assert isinstance(transform, float)
    assert transform == 12 / 7
