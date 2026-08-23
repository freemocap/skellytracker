"""Tests for skellytracker.core.sidecar.runtime — normalization dispatch and
OnnxModelSpec construction from a resolved sidecar artifact.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from skellytracker.core.sessions.model_registry import resolve_model_path
from skellytracker.core.sidecar.model import SidecarModel
from skellytracker.core.sidecar.runtime import (
    build_normalization_fn,
    resolve_normalization_mode,
    sidecar_model_spec,
)


def _detector_sidecar_dict(**input_overrides) -> dict:
    input_spec = {
        "name": "images",
        "dtype": {"fp32": "float32"},
        "normalization": "unit_float",
        "resize": {"method": "letterbox"},
    }
    input_spec.update(input_overrides)
    return {
        "schema_version": "v2024.09.1019",
        "model_id": "toy",
        "display_name": "Toy",
        "role": ["object_detector"],
        "input": input_spec,
        "outputs": [
            {
                "name": "output0",
                "dtype": {"fp32": "float32"},
                "semantic": "detections",
                "fields": ["x1", "y1", "x2", "y2", "score"],
            }
        ],
        "decode": {"box_format": "xyxy"},
        "sizes": {
            "nano": {
                "input": {
                    "shape": [-1, 3, 640, 640],
                    "resize": {"target_size": [640, 640]},
                },
                "onnx": {
                    "batch_artifacts": {
                        "2": {
                            "precision_artifacts": {
                                "fp32": {"filename": "toy-nano_b2_fp32.onnx"}
                            }
                        }
                    }
                },
            }
        },
    }


class TestNormalizationDispatch:
    def test_none_mode_passes_through(self):
        sidecar = SidecarModel.model_validate(
            _detector_sidecar_dict(normalization="none")
        )
        fn = build_normalization_fn(sidecar.input, "fp32")
        img = np.full((2, 2, 3), 200, dtype=np.uint8)
        np.testing.assert_allclose(fn(img), img.astype(np.float32))

    def test_unit_float_mode_divides_by_255(self):
        sidecar = SidecarModel.model_validate(
            _detector_sidecar_dict(normalization="unit_float")
        )
        fn = build_normalization_fn(sidecar.input, "fp32")
        img = np.full((1, 1, 3), 255, dtype=np.uint8)
        np.testing.assert_allclose(fn(img), np.ones((1, 1, 3), dtype=np.float32))

    def test_imagenet_bgr_mode_matches_known_constants(self):
        sidecar = SidecarModel.model_validate(
            _detector_sidecar_dict(normalization="imagenet_bgr")
        )
        fn = build_normalization_fn(sidecar.input, "fp32")
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        expected = (0.0 - np.array([123.675, 116.28, 103.53])) / np.array(
            [58.395, 57.12, 57.375]
        )
        np.testing.assert_allclose(fn(img)[0, 0], expected, rtol=1e-5)

    def test_custom_mode(self):
        data = _detector_sidecar_dict(
            normalization={
                "mode": "custom",
                "scale": 0.5,
                "mean": [1.0, 1.0, 1.0],
                "std": [2.0, 2.0, 2.0],
            }
        )
        sidecar = SidecarModel.model_validate(data)
        fn = build_normalization_fn(sidecar.input, "fp32")
        img = np.full((1, 1, 3), 10, dtype=np.uint8)
        # (10 * 0.5 - 1) / 2 = 2.0
        np.testing.assert_allclose(fn(img)[0, 0], [2.0, 2.0, 2.0])

    def test_normalization_by_precision_overrides_top_level(self):
        data = _detector_sidecar_dict(
            normalization="imagenet_bgr", normalization_by_precision={"int8": "none"}
        )
        sidecar = SidecarModel.model_validate(data)
        assert resolve_normalization_mode(sidecar.input, "int8") == "none"
        assert resolve_normalization_mode(sidecar.input, "fp32") == "imagenet_bgr"


def _detector_sidecar_dict_with_url(**input_overrides) -> dict:
    data = _detector_sidecar_dict(**input_overrides)
    data["sizes"]["nano"]["onnx"]["batch_artifacts"]["2"]["precision_artifacts"][
        "fp32"
    ]["url"] = "https://example.com/toy-nano_b2_fp32.zip"
    data["sizes"]["nano"]["onnx"]["batch_artifacts"]["2"]["precision_artifacts"][
        "fp32"
    ]["url_sha256"] = ("a" * 64)
    return data


class TestSidecarModelSpec:
    """`sidecar_model_spec()` is pure: it never touches the filesystem or
    network. Resolution (download, extraction, SHA-256 verification) is
    deferred entirely to whatever later calls `resolve_model_path()` on the
    returned spec's `source` — normally `OnnxSession.create()`.
    """

    def test_local_artifact_builds_onnx_model_spec(self, tmp_path: Path):
        sidecar_dir = tmp_path
        sidecar = SidecarModel.model_validate(_detector_sidecar_dict())
        spec = sidecar_model_spec(
            sidecar,
            size="nano",
            batch_key="2",
            precision="fp32",
            name="toy-nano",
            sidecar_dir=sidecar_dir,
        )

        assert spec.name == "toy-nano"
        assert spec.input_size == (640, 640)
        assert Path(spec.source.local_path).name == "toy-nano_b2_fp32.onnx"
        assert spec.expected_filename == "toy-nano_b2_fp32.onnx"

    def test_missing_local_artifact_is_not_checked_until_resolved(self, tmp_path: Path):
        # No file written to tmp_path — building the spec must not touch the
        # filesystem to validate it exists; only resolving it later does.
        sidecar = SidecarModel.model_validate(_detector_sidecar_dict())
        spec = sidecar_model_spec(
            sidecar,
            size="nano",
            batch_key="2",
            precision="fp32",
            name="toy-nano",
            sidecar_dir=tmp_path,
        )
        with pytest.raises(FileNotFoundError):
            resolve_model_path(spec.source)

    def test_url_artifact_performs_no_network_io(self, tmp_path: Path):
        sidecar = SidecarModel.model_validate(_detector_sidecar_dict_with_url())
        with patch(
            "skellytracker.core.sessions.model_registry.requests.get"
        ) as mock_get:
            spec = sidecar_model_spec(
                sidecar,
                size="nano",
                batch_key="2",
                precision="fp32",
                name="toy-nano",
                sidecar_dir=tmp_path,
            )
        mock_get.assert_not_called()
        assert spec.source.url == "https://example.com/toy-nano_b2_fp32.zip"

    def test_url_artifact_populates_expected_filename_and_sha256(self, tmp_path: Path):
        sidecar = SidecarModel.model_validate(_detector_sidecar_dict_with_url())
        spec = sidecar_model_spec(
            sidecar,
            size="nano",
            batch_key="2",
            precision="fp32",
            name="toy-nano",
            sidecar_dir=tmp_path,
        )
        assert spec.expected_filename == "toy-nano_b2_fp32.onnx"
        assert spec.expected_sha256 == "a" * 64

    def test_name_is_caller_supplied_not_derived(self, tmp_path: Path):
        sidecar = SidecarModel.model_validate(_detector_sidecar_dict())
        spec = sidecar_model_spec(
            sidecar,
            size="nano",
            batch_key="2",
            precision="fp32",
            name="whatever-i-want",
            sidecar_dir=tmp_path,
        )
        assert spec.name == "whatever-i-want"
