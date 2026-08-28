"""Tests for generic image preprocessing: letterbox resize and normalization."""

from __future__ import annotations

import numpy as np
import pytest

from skellytracker.core.detectors.processing.image_preprocessing import (
    build_normalization_fn,
    letterbox_preprocess,
    preprocess_image,
    resolve_normalization_mode,
)
from skellytracker.core.detectors.processing.affine_crop import (
    bbox_xyxy2cs,
    top_down_affine,
)
from skellytracker.core.sidecar.model import InputSpec

# ---------------------------------------------------------------------------
# letterbox_preprocess
# ---------------------------------------------------------------------------


class TestLetterboxPreprocess:
    def test_output_shape_matches_target(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, _ = letterbox_preprocess(img, (640, 640))
        assert padded.shape == (640, 640, 3)

    def test_ratio_for_square_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, ratio = letterbox_preprocess(img, (200, 200))
        assert ratio == pytest.approx(2.0)

    def test_ratio_limited_by_shorter_side(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, ratio = letterbox_preprocess(img, (640, 640))
        assert ratio == pytest.approx(640 / 640)

    def test_ratio_limited_by_height(self):
        img = np.zeros((800, 400, 3), dtype=np.uint8)
        _, ratio = letterbox_preprocess(img, (640, 640))
        assert ratio == pytest.approx(640 / 800)

    def test_padding_color_is_114_by_default(self):
        # 100h × 50w image → ratio=2.0, resized to 200h × 100w.
        # Columns 100–200 are untouched padding.
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        padded, _ = letterbox_preprocess(img, (200, 200))
        assert padded[100, 150, 0] == 114

    def test_pad_value_is_configurable(self):
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        padded, _ = letterbox_preprocess(img, (200, 200), pad_value=0)
        assert padded[100, 150, 0] == 0

    def test_output_dtype_is_uint8(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        padded, _ = letterbox_preprocess(img, (416, 416))
        assert padded.dtype == np.uint8


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------


def _input_spec(**overrides) -> InputSpec:
    fields = {
        "name": "images",
        "dtype": {"fp32": "float32"},
        "normalization": "unit_float",
    }
    fields.update(overrides)
    return InputSpec.model_validate(fields)


class TestNormalizationDispatch:
    def test_none_mode_passes_through(self):
        spec = _input_spec(normalization="none")
        fn = build_normalization_fn(spec, "fp32")
        img = np.full((2, 2, 3), 200, dtype=np.uint8)
        np.testing.assert_allclose(fn(img), img.astype(np.float32))

    def test_unit_float_mode_divides_by_255(self):
        spec = _input_spec(normalization="unit_float")
        fn = build_normalization_fn(spec, "fp32")
        img = np.full((1, 1, 3), 255, dtype=np.uint8)
        np.testing.assert_allclose(fn(img), np.ones((1, 1, 3), dtype=np.float32))

    def test_imagenet_bgr_mode_matches_known_constants(self):
        spec = _input_spec(normalization="imagenet_bgr")
        fn = build_normalization_fn(spec, "fp32")
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        expected = (0.0 - np.array([123.675, 116.28, 103.53])) / np.array(
            [58.395, 57.12, 57.375]
        )
        np.testing.assert_allclose(fn(img)[0, 0], expected, rtol=1e-5)

    def test_custom_mode(self):
        spec = _input_spec(
            normalization={
                "mode": "custom",
                "scale": 0.5,
                "mean": [1.0, 1.0, 1.0],
                "std": [2.0, 2.0, 2.0],
            }
        )
        fn = build_normalization_fn(spec, "fp32")
        img = np.full((1, 1, 3), 10, dtype=np.uint8)
        # (10 * 0.5 - 1) / 2 = 2.0
        np.testing.assert_allclose(fn(img)[0, 0], [2.0, 2.0, 2.0])

    def test_normalization_by_precision_overrides_top_level(self):
        spec = _input_spec(
            normalization="imagenet_bgr", normalization_by_precision={"int8": "none"}
        )
        assert resolve_normalization_mode(spec, "int8") == "none"
        assert resolve_normalization_mode(spec, "fp32") == "imagenet_bgr"


# ---------------------------------------------------------------------------
# preprocess_image — affine_person_crop
# ---------------------------------------------------------------------------


def _affine_input_spec() -> InputSpec:
    return InputSpec.model_validate(
        {
            "name": "input",
            "dtype": {"fp32": "float32"},
            "normalization": "imagenet_bgr",
            "resize": {
                "method": "affine_person_crop",
                "crop_policy": {"expand_ratio": 1.25, "maintain_aspect_ratio": True},
            },
        }
    )


class TestPreprocessImageAffinePersonCrop:
    def test_output_tensor_shape(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, _ = preprocess_image(img, (256, 192), _affine_input_spec())
        assert tensor.shape == (3, 256, 192)
        assert tensor.dtype == np.float32

    def test_resize_meta_shapes(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, (center, scale) = preprocess_image(img, (256, 192), _affine_input_spec())
        assert center.shape == (2,)
        assert scale.shape == (2,)

    def test_full_image_bbox_matches_manual_affine_crop(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        target_size = (256, 192)  # (H, W)

        tensor, (center, scale) = preprocess_image(
            img, target_size, _affine_input_spec()
        )

        h, w = img.shape[:2]
        expected_bbox = np.array([0.0, 0.0, float(w), float(h)], dtype=np.float64)
        expected_center, expected_scale_pre = bbox_xyxy2cs(expected_bbox, padding=1.25)
        target_h, target_w = target_size
        expected_warped, expected_scale = top_down_affine(
            (target_w, target_h), expected_scale_pre, expected_center, img
        )

        np.testing.assert_allclose(center, expected_center)
        np.testing.assert_allclose(scale, expected_scale)

        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        expected_normalized = (expected_warped.astype(np.float32) - mean) / std
        expected_tensor = np.ascontiguousarray(
            expected_normalized.transpose(2, 0, 1).astype(np.float32)
        )
        np.testing.assert_allclose(tensor, expected_tensor, rtol=1e-5, atol=1e-5)

    def test_maintain_aspect_ratio_false_raises(self):
        spec = InputSpec.model_validate(
            {
                "name": "input",
                "dtype": {"fp32": "float32"},
                "resize": {
                    "method": "affine_person_crop",
                    "crop_policy": {
                        "expand_ratio": 1.25,
                        "maintain_aspect_ratio": False,
                    },
                },
            }
        )
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(NotImplementedError):
            preprocess_image(img, (256, 192), spec)
