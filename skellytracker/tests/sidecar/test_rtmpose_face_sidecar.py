"""Tests for the RTMPose face sidecar (M4b): loads the real rtmpose-face.yaml
and checks post-migration detector output against a pre-migration golden
fixture, compared by keypoint NAME.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from skellytracker.core.sidecar.loader import load_sidecar

_FACE_DIR = (
    Path(__file__).resolve().parents[2]
    / "core"
    / "detectors"
    / "keypoint_detectors"
    / "rtmpose"
    / "face"
)
_GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"

_MODEL_NAMES = ["rtmpose-m_256x256"]


class TestRTMPoseFaceSidecarValidation:
    def test_loads_and_validates(self):
        sidecar = load_sidecar(_FACE_DIR / "rtmpose-face.yaml")
        assert sidecar.model_id == "rtmpose-face"
        assert sidecar.role == ["pose_estimator"]

    def test_has_size(self):
        sidecar = load_sidecar(_FACE_DIR / "rtmpose-face.yaml")
        assert set(sidecar.sizes) == set(_MODEL_NAMES)

    def test_tracked_points_count(self):
        sidecar = load_sidecar(_FACE_DIR / "rtmpose-face.yaml")
        assert len(sidecar.pose.tracked_points) == 106
        assert len(set(sidecar.pose.tracked_points)) == 106
        assert sidecar.pose.tracked_points[0] == "face_0000"
        assert sidecar.pose.tracked_points[-1] == "face_0105"

    def test_no_canonical_mapping_yet(self):
        # Pre-existing gap: no canonical-mapping YAML exists for face today
        # (see specs/sidecar-implementation-plan.md M4b). Preserved as-is.
        sidecar = load_sidecar(_FACE_DIR / "rtmpose-face.yaml")
        assert sidecar.pose.canonical_mapping is None

    def test_target_size_matches_pre_migration_constant(self):
        sidecar = load_sidecar(_FACE_DIR / "rtmpose-face.yaml")
        assert tuple(
            sidecar.resolved_size("rtmpose-m_256x256").input.resize.target_size
        ) == (256, 256)


@pytest.mark.parametrize("model_name", _MODEL_NAMES)
class TestRTMPoseFaceParityAgainstGoldenFixture:
    def test_detect_matches_golden_fixture_by_name(self, model_name, test_image):
        pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
        golden_path = (
            _GOLDEN_DIR / f"rtmpose_face_{model_name.replace('-', '_')}_detect.json"
        )
        if not golden_path.exists():
            pytest.skip(f"no golden fixture at {golden_path}")
        golden = json.loads(golden_path.read_text())

        from skellytracker.core.detectors.keypoint_detectors.rtmpose.face.rtmpose_face_detector import (
            RTMPoseFaceDetector,
            RTMPoseFaceDetectorConfig,
        )
        from skellytracker.core.sessions.onnx_session import (
            OnnxSession,
            OnnxSessionConfig,
        )

        config = OnnxSessionConfig(
            batch_size=1, models=[RTMPoseFaceDetector.model_spec(model_name)]
        )
        session = OnnxSession.create(config)
        try:
            detector = RTMPoseFaceDetector.create(
                RTMPoseFaceDetectorConfig(model_name=model_name), session
            )
            kpts = detector.detect(test_image)
        finally:
            session.close()

        assert set(golden.keys()) <= set(kpts.names)

        seen = 0
        for name, xyz in zip(kpts.names, kpts.xyz, strict=True):
            if name not in golden:
                assert np.isnan(xyz[0]) or np.isnan(xyz[1])
                continue
            seen += 1
            expected = golden[name]
            actual = np.array([xyz[0], xyz[1]])
            expected_arr = np.array([expected["x"], expected["y"]])
            assert np.allclose(
                actual, expected_arr, atol=1e-3
            ), f"{name}: actual={actual}, expected={expected_arr}"
        assert seen == len(golden)


def _fake_response(body: bytes) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.headers = {"content-length": str(len(body))}
    resp.iter_content = MagicMock(return_value=[body])
    return resp


class TestRTMPoseFaceModelSpecVerification:
    def test_model_spec_performs_no_network_io(self):
        from skellytracker.core.detectors.keypoint_detectors.rtmpose.face.rtmpose_face_detector import (
            RTMPoseFaceDetector,
        )

        with patch(
            "skellytracker.core.sessions.model_registry.requests.get"
        ) as mock_get:
            for model_name in _MODEL_NAMES:
                RTMPoseFaceDetector.model_spec(model_name)
        mock_get.assert_not_called()

    def test_model_spec_populates_expected_filename_and_sha256(self):
        from skellytracker.core.detectors.keypoint_detectors.rtmpose.face.rtmpose_face_detector import (
            RTMPoseFaceDetector,
        )

        sidecar = load_sidecar(_FACE_DIR / "rtmpose-face.yaml")

        for model_name in _MODEL_NAMES:
            spec = RTMPoseFaceDetector.model_spec(model_name)
            artifact = (
                sidecar.resolved_size(model_name)
                .onnx.batch_artifacts["1"]
                .precision_artifacts["fp32"]
            )
            assert spec.expected_filename == artifact.filename
            assert spec.expected_sha256 == artifact.url_sha256
            assert spec.expected_sha256 is not None

    def test_resolve_model_path_verifies_matching_sha256(self, tmp_path):
        import hashlib
        import io
        import zipfile

        from skellytracker.core.detectors.keypoint_detectors.rtmpose.face.rtmpose_face_detector import (
            RTMPoseFaceDetector,
        )
        from skellytracker.core.sessions.model_registry import resolve_model_path

        spec = RTMPoseFaceDetector.model_spec("rtmpose-m_256x256")
        onnx_bytes = b"fake-onnx-inside-zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("some/nested/end2end.onnx", onnx_bytes)
        zip_bytes = buf.getvalue()
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
        assert path.read_bytes() == onnx_bytes
        assert path.name == spec.expected_filename

    def test_resolve_model_path_rejects_mismatched_sha256(self, tmp_path):
        from skellytracker.core.detectors.keypoint_detectors.rtmpose.face.rtmpose_face_detector import (
            RTMPoseFaceDetector,
        )
        from skellytracker.core.sessions.model_registry import (
            ModelIntegrityError,
            resolve_model_path,
        )

        spec = RTMPoseFaceDetector.model_spec("rtmpose-m_256x256")
        body = b"fake-zip-bytes-for-face"

        with patch(
            "skellytracker.core.sessions.model_registry.requests.get",
            return_value=_fake_response(body),
        ):
            with pytest.raises(ModelIntegrityError):
                resolve_model_path(
                    spec.source,
                    cache_dir=tmp_path,
                    expected_filename=spec.expected_filename,
                    expected_sha256=spec.expected_sha256,
                )
        assert not (tmp_path / spec.expected_filename).exists()
