import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from skellytracker.core.sidecar.model import DetectionDecodeSpec, InputSpec, ResizeSpec
from skellytracker.core.detectors.processing.image_preprocessing import (
    build_normalization_fn,
    preprocess_image,
)


def image_spec(**changes):
    return InputSpec.model_validate(
        dict(name="input", dtype={"fp32": "float32"}, **changes)
    )


def test_rgb_normalization_reorders_bgr_input():
    image = np.array([[[10, 20, 30]]], dtype=np.uint8)
    spec = image_spec(normalization={"mode": "custom", "color_format": "rgb"})
    np.testing.assert_array_equal(
        build_normalization_fn(spec, "fp32")(image), [[[30.0, 20.0, 10.0]]]
    )


def test_affine_interpolation_is_honored():
    image = np.random.default_rng(3).integers(0, 256, (17, 23, 3), dtype=np.uint8)
    spec = image_spec(
        normalization="none",
        resize={"method": "affine_person_crop", "interpolation": "nearest"},
    )
    actual, (center, scale) = preprocess_image(image, (20, 16), spec)
    from skellytracker.core.detectors.processing.affine_crop import get_warp_matrix

    transform = get_warp_matrix(center, scale, 0.0, output_size=(16, 20))
    expected = cv2.warpAffine(image, transform, (16, 20), flags=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(
        actual, expected.transpose(2, 0, 1).astype(np.float32)
    )


@pytest.mark.parametrize(
    "fields",
    [
        dict(method="fit", pad_position="center"),
        dict(method="stretch", pad_value=114),
        dict(method="letterbox", pad_position="undeclared"),
        dict(method="letterbox", rounding="undeclared"),
    ],
)
def test_invalid_resize_combinations_are_rejected(fields):
    with pytest.raises(ValidationError):
        ResizeSpec.model_validate(fields)


def test_unimplemented_selection_contract_is_rejected():
    with pytest.raises(ValidationError):
        DetectionDecodeSpec.model_validate({"selection": "top_k"})


def test_quantized_input_is_not_silently_converted_to_float32():
    spec = InputSpec(name="input", dtype={"int8": "uint8"}, normalization="none")
    with pytest.raises(NotImplementedError, match="float32"):
        build_normalization_fn(spec, "int8")


def test_extreme_aspect_ratio_letterbox_remains_nonempty():
    spec = image_spec(normalization="none", resize={"method": "letterbox"})
    tensor, ratio = preprocess_image(
        np.zeros((1, 10000, 3), dtype=np.uint8), (20, 20), spec
    )
    assert tensor.shape == (3, 20, 20)
    assert ratio > 0
