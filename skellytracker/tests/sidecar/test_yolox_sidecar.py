"""Tests for the YOLOX sidecar (M3): loads the real yolox.yaml and checks
post-migration detector output against a pre-migration golden fixture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from skellytracker.core.sidecar.loader import load_sidecar

_YOLOX_DIR = (
    Path(__file__).resolve().parents[2]
    / "core"
    / "detectors"
    / "object_detectors"
    / "yolox"
)
_GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"


class TestYoloxSidecarValidation:
    def test_loads_and_validates(self):
        sidecar = load_sidecar(_YOLOX_DIR / "yolox.yaml")
        assert sidecar.model_id == "yolox"
        assert sidecar.role == ["object_detector"]

    def test_has_both_sizes(self):
        sidecar = load_sidecar(_YOLOX_DIR / "yolox.yaml")
        assert set(sidecar.sizes) == {"yolox-tiny", "yolox-m"}

    def test_decode_contract(self):
        sidecar = load_sidecar(_YOLOX_DIR / "yolox.yaml")
        assert sidecar.decode.box_format == "xyxy"
        assert sidecar.decode.requires_nms is False

    def test_target_sizes_match_pre_migration_constants(self):
        sidecar = load_sidecar(_YOLOX_DIR / "yolox.yaml")
        assert tuple(sidecar.resolved_size("yolox-m").input.resize.target_size) == (
            640,
            640,
        )
        assert tuple(sidecar.resolved_size("yolox-tiny").input.resize.target_size) == (
            416,
            416,
        )


@pytest.mark.parametrize("model_name", ["yolox-m", "yolox-tiny"])
class TestYoloxParityAgainstGoldenFixture:
    def test_detect_matches_golden_fixture(self, model_name, test_image):
        pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
        golden_path = _GOLDEN_DIR / f"{model_name.replace('-', '_')}_detect.json"
        if not golden_path.exists():
            pytest.skip(f"no golden fixture at {golden_path}")
        golden = json.loads(golden_path.read_text())

        from skellytracker.core.detectors.object_detectors.yolox.yolox_person_detector import (
            YoloxPersonDetector,
            YoloxPersonDetectorConfig,
        )
        from skellytracker.core.sessions.onnx_session import (
            OnnxSession,
            OnnxSessionConfig,
        )

        config = OnnxSessionConfig(
            batch_size=1, models=[YoloxPersonDetector.model_spec(model_name)]
        )
        session = OnnxSession.create(config)
        try:
            detector = YoloxPersonDetector.create(
                YoloxPersonDetectorConfig(model_name=model_name, max_detections=None),
                session,
            )
            results = detector.detect(test_image)
        finally:
            session.close()

        assert len(results) == len(golden)
        for bb, expected in zip(results, golden, strict=True):
            actual = np.array([bb.x1, bb.y1, bb.x2, bb.y2, bb.confidence])
            expected_arr = np.array(
                [
                    expected["x1"],
                    expected["y1"],
                    expected["x2"],
                    expected["y2"],
                    expected["confidence"],
                ]
            )
            assert np.allclose(actual, expected_arr, atol=1e-4)


def _fake_response(body: bytes) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.headers = {"content-length": str(len(body))}
    resp.iter_content = MagicMock(return_value=[body])
    return resp


class TestYoloxModelSpecVerification:
    """model_spec() must stay lazy (no network I/O) and must correctly
    surface the sidecar's filename/sha256 so OnnxSession.create() can verify
    downloads. See specs/sidecar-implementation-plan.md's M3 status note.
    """

    def test_model_spec_performs_no_network_io(self):
        from skellytracker.core.detectors.object_detectors.yolox.yolox_person_detector import (
            YoloxPersonDetector,
        )

        with patch(
            "skellytracker.core.sessions.model_registry.requests.get"
        ) as mock_get:
            YoloxPersonDetector.model_spec("yolox-m")
            YoloxPersonDetector.model_spec("yolox-tiny")
        mock_get.assert_not_called()

    def test_model_spec_populates_expected_filename_and_sha256(self):
        from skellytracker.core.detectors.object_detectors.yolox.yolox_person_detector import (
            YoloxPersonDetector,
        )

        sidecar = load_sidecar(_YOLOX_DIR / "yolox.yaml")

        for model_name in ("yolox-m", "yolox-tiny"):
            spec = YoloxPersonDetector.model_spec(model_name)
            artifact = (
                sidecar.resolved_size(model_name)
                .onnx.batch_artifacts["1"]
                .precision_artifacts["fp32"]
            )
            assert spec.expected_filename == artifact.filename
            assert spec.expected_sha256 == artifact.url_sha256
            assert spec.expected_sha256 is not None

    def test_resolve_model_path_verifies_matching_sha256(self, tmp_path):
        import io
        import zipfile

        from skellytracker.core.detectors.object_detectors.yolox.yolox_person_detector import (
            YoloxPersonDetector,
        )
        from skellytracker.core.sessions.model_registry import resolve_model_path

        spec = YoloxPersonDetector.model_spec("yolox-tiny")
        onnx_bytes = b"fake-onnx-inside-zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("upstream_internal_name.onnx", onnx_bytes)
        zip_bytes = buf.getvalue()
        # Verify against the *actual* bytes served (a fake zip), not the real
        # sidecar sha256 — this test only checks the "hashes match" path.
        digest = hashlib.sha256(zip_bytes).hexdigest()

        with patch(
            "skellytracker.core.sessions.model_registry.requests.get",
            return_value=_fake_response(zip_bytes),
        ):
            path = resolve_model_path(
                spec.source,
                cache_dir=tmp_path,
                expected_filename=spec.expected_filename,
                expected_sha256=digest,
            )
        assert path.name == spec.expected_filename
        assert path.read_bytes() == onnx_bytes

    def test_resolve_model_path_rejects_mismatched_sha256(self, tmp_path):
        from skellytracker.core.detectors.object_detectors.yolox.yolox_person_detector import (
            YoloxPersonDetector,
        )
        from skellytracker.core.sessions.model_registry import (
            ModelIntegrityError,
            resolve_model_path,
        )

        spec = YoloxPersonDetector.model_spec("yolox-tiny")
        body = b"fake-zip-bytes-for-yolox-tiny"

        with patch(
            "skellytracker.core.sessions.model_registry.requests.get",
            return_value=_fake_response(body),
        ):
            with pytest.raises(ModelIntegrityError):
                resolve_model_path(
                    spec.source,
                    cache_dir=tmp_path,
                    expected_filename=spec.expected_filename,
                    expected_sha256=spec.expected_sha256,  # the real (mismatched) sidecar digest
                )
        assert not (tmp_path / spec.expected_filename).exists()


class TestOnnxSessionThreadsVerificationFields:
    def test_create_passes_expected_filename_and_sha256_to_resolve(self, tmp_path):
        pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
        import onnxruntime as ort

        from skellytracker.core.sessions.model_registry import ModelSource
        from skellytracker.core.sessions.onnx_session import (
            OnnxModelSpec,
            OnnxSession,
            OnnxSessionConfig,
        )

        local_file = tmp_path / "stub.onnx"
        local_file.write_bytes(b"not-a-real-onnx-model")
        spec = OnnxModelSpec(
            name="stub",
            source=ModelSource(local_path=str(local_file)),
            input_size=(64, 64),
            expected_filename="stub-expected.onnx",
            expected_sha256="deadbeef",
        )
        config = OnnxSessionConfig(batch_size=1, models=[spec])
        fake_ort_session = MagicMock(spec=ort.InferenceSession)

        with patch(
            "skellytracker.core.sessions.onnx_session.resolve_model_path",
            return_value=local_file,
        ) as mock_resolve:
            with patch(
                "skellytracker.core.sessions.onnx_session.build_tuned_ort_session",
                return_value=fake_ort_session,
            ):
                OnnxSession.create(config)

        mock_resolve.assert_called_once_with(
            spec.source,
            expected_filename="stub-expected.onnx",
            expected_sha256="deadbeef",
        )
