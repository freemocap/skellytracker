"""Validate the sidecar and compare real inference with the original math on the same backend.

Upstream golden JSON is retained as historical evidence; its provider and
precision are unspecified, so it is not a strict numerical oracle here.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from skellytracker.tests.sidecar.model_fixtures import cached_model_spec, assert_original_pose_processing

from skellytracker.core.sidecar.loader import load_sidecar

_BODY_DIR = (
    Path(__file__).resolve().parents[2]
    / "core"
    / "detectors"
    / "keypoint_detectors"
    / "rtmpose"
    / "body"
)

_MODEL_NAMES = ["rtmpose-s_256x192", "rtmpose-m_256x192"]


class TestRTMPoseBodySidecarValidation:
    def test_loads_and_validates(self):
        sidecar = load_sidecar(_BODY_DIR / "rtmpose-body.yaml")
        assert sidecar.model_id == "rtmpose-body"
        assert sidecar.role == ["pose_estimator"]

    def test_has_both_sizes(self):
        sidecar = load_sidecar(_BODY_DIR / "rtmpose-body.yaml")
        assert set(sidecar.sizes) == set(_MODEL_NAMES)

    def test_tracked_points_count(self):
        sidecar = load_sidecar(_BODY_DIR / "rtmpose-body.yaml")
        assert len(sidecar.pose.tracked_points) == 17
        assert len(set(sidecar.pose.tracked_points)) == 17
        assert sidecar.pose.tracked_points[0] == "nose"
        assert sidecar.pose.tracked_points[-1] == "right_ankle"

    def test_canonical_mapping_direct_passthrough(self):
        sidecar = load_sidecar(_BODY_DIR / "rtmpose-body.yaml")
        mapping = sidecar.pose.canonical_mapping
        assert mapping["nose"] == "nose"
        assert mapping["left_shoulder"] == "left_shoulder"

    def test_target_sizes_match_pre_migration_constants(self):
        sidecar = load_sidecar(_BODY_DIR / "rtmpose-body.yaml")
        assert tuple(
            sidecar.resolved_size("rtmpose-s_256x192").input.resize.target_size
        ) == (256, 192)
        assert tuple(
            sidecar.resolved_size("rtmpose-m_256x192").input.resize.target_size
        ) == (256, 192)


@pytest.mark.parametrize("model_name", _MODEL_NAMES)
class TestRTMPoseBodyParityAgainstOriginalProcessing:
    def test_detect_matches_original_processing_by_name(self, model_name, test_image):
        pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
        from skellytracker.core.detectors.keypoint_detectors.rtmpose.body.rtmpose_body_detector import (
            RTMPoseBodyDetector,
            RTMPoseBodyDetectorConfig,
        )
        from skellytracker.core.sessions.onnx_session import (
            OnnxSession,
            OnnxSessionConfig,
        )

        config = OnnxSessionConfig(
            batch_size=1, execution_provider="cpu", fp16=False, models=[cached_model_spec(RTMPoseBodyDetector.model_spec(model_name))]
        )
        session = OnnxSession.create(config)
        try:
            detector = RTMPoseBodyDetector.create(
                RTMPoseBodyDetectorConfig(model_name=model_name), session
            )
            assert_original_pose_processing(detector, test_image, load_sidecar(_BODY_DIR / "rtmpose-body.yaml").pose.tracked_points)
        finally:
            session.close()



def _fake_response(body: bytes) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.headers = {"content-length": str(len(body))}
    resp.iter_content = MagicMock(return_value=[body])
    return resp


class TestRTMPoseBodyModelSpecVerification:
    def test_model_spec_performs_no_network_io(self):
        from skellytracker.core.detectors.keypoint_detectors.rtmpose.body.rtmpose_body_detector import (
            RTMPoseBodyDetector,
        )

        with patch(
            "skellytracker.core.sessions.model_registry.requests.get"
        ) as mock_get:
            for model_name in _MODEL_NAMES:
                RTMPoseBodyDetector.model_spec(model_name)
        mock_get.assert_not_called()

    def test_model_spec_populates_expected_filename_and_sha256(self):
        from skellytracker.core.detectors.keypoint_detectors.rtmpose.body.rtmpose_body_detector import (
            RTMPoseBodyDetector,
        )

        sidecar = load_sidecar(_BODY_DIR / "rtmpose-body.yaml")

        for model_name in _MODEL_NAMES:
            spec = RTMPoseBodyDetector.model_spec(model_name)
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

        from skellytracker.core.detectors.keypoint_detectors.rtmpose.body.rtmpose_body_detector import (
            RTMPoseBodyDetector,
        )
        from skellytracker.core.sessions.model_registry import resolve_model_path

        spec = RTMPoseBodyDetector.model_spec("rtmpose-s_256x192")
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
        from skellytracker.core.detectors.keypoint_detectors.rtmpose.body.rtmpose_body_detector import (
            RTMPoseBodyDetector,
        )
        from skellytracker.core.sessions.model_registry import (
            ModelIntegrityError,
            resolve_model_path,
        )

        spec = RTMPoseBodyDetector.model_spec("rtmpose-s_256x192")
        body = b"fake-zip-bytes-for-body"

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
