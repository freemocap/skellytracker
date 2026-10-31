"""Validate the sidecar and compare real inference with the original math on the same backend.

Upstream golden JSON is retained as historical evidence; its provider and
precision are unspecified, so it is not a strict numerical oracle here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from skellytracker.tests.sidecar.model_fixtures import cached_model_spec, assert_original_pose_processing

from skellytracker.core.sidecar.loader import load_sidecar

_RTMW_DIR = (
    Path(__file__).resolve().parents[2]
    / "core"
    / "detectors"
    / "keypoint_detectors"
    / "rtmw"
    / "wholebody"
)

_MODEL_NAMES = ["rtmw-l-m_256x192", "rtmw-x-l_256x192", "rtmw-x-l_384x288"]


class TestRTMWSidecarValidation:
    def test_loads_and_validates(self):
        sidecar = load_sidecar(_RTMW_DIR / "rtmw-wholebody.yaml")
        assert sidecar.model_id == "rtmw-wholebody"
        assert sidecar.role == ["pose_estimator"]

    def test_has_all_three_sizes(self):
        sidecar = load_sidecar(_RTMW_DIR / "rtmw-wholebody.yaml")
        assert set(sidecar.sizes) == set(_MODEL_NAMES)

    def test_tracked_points_count(self):
        sidecar = load_sidecar(_RTMW_DIR / "rtmw-wholebody.yaml")
        assert len(sidecar.pose.tracked_points) == 133
        assert len(set(sidecar.pose.tracked_points)) == 133

    def test_tracked_points_native_model_order(self):
        sidecar = load_sidecar(_RTMW_DIR / "rtmw-wholebody.yaml")
        tp = sidecar.pose.tracked_points
        # native order: body(0-22) + face(23-90) + left_hand(91-111) + right_hand(112-132)
        assert tp[0] == "nose"
        assert tp[22] == "right_heel"
        assert tp[23] == "face_0000"
        assert tp[90] == "face_0067"
        assert tp[91] == "left_hand_root"
        assert tp[111] == "left_hand_pinky_finger4"
        assert tp[112] == "right_hand_root"
        assert tp[132] == "right_hand_pinky_finger4"

    def test_canonical_mapping_prefix_expansion(self):
        sidecar = load_sidecar(_RTMW_DIR / "rtmw-wholebody.yaml")
        mapping = sidecar.pose.canonical_mapping
        assert mapping["right_hand_wrist"] == "right_hand_root"
        assert mapping["left_hand_wrist"] == "left_hand_root"
        assert mapping["nose"] == "nose"  # from the composed body mapping

    def test_overlay_groups_has_four_entries(self):
        sidecar = load_sidecar(_RTMW_DIR / "rtmw-wholebody.yaml")
        assert set(sidecar.overlay.groups) == {
            "right_hand",
            "left_hand",
            "face",
            "body",
        }

    def test_target_sizes_match_pre_migration_constants(self):
        sidecar = load_sidecar(_RTMW_DIR / "rtmw-wholebody.yaml")
        assert tuple(
            sidecar.resolved_size("rtmw-l-m_256x192").input.resize.target_size
        ) == (256, 192)
        assert tuple(
            sidecar.resolved_size("rtmw-x-l_256x192").input.resize.target_size
        ) == (256, 192)
        assert tuple(
            sidecar.resolved_size("rtmw-x-l_384x288").input.resize.target_size
        ) == (384, 288)


@pytest.mark.parametrize("model_name", _MODEL_NAMES)
class TestRTMWParityAgainstOriginalProcessing:
    def test_detect_matches_original_processing_by_name(self, model_name, test_image):
        pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
        from skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.rtmw_wholebody_detector import (
            RTMWWholebodyDetector,
            RTMWWholebodyDetectorConfig,
        )
        from skellytracker.core.sessions.onnx_session import (
            OnnxSession,
            OnnxSessionConfig,
        )

        config = OnnxSessionConfig(
            batch_size=1, execution_provider="cpu", fp16=False, models=[cached_model_spec(RTMWWholebodyDetector.model_spec(model_name))]
        )
        session = OnnxSession.create(config)
        try:
            detector = RTMWWholebodyDetector.create(
                RTMWWholebodyDetectorConfig(model_name=model_name), session
            )
            assert_original_pose_processing(detector, test_image, load_sidecar(_RTMW_DIR / "rtmw-wholebody.yaml").pose.tracked_points)
        finally:
            session.close()



def _fake_response(body: bytes) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.headers = {"content-length": str(len(body))}
    resp.iter_content = MagicMock(return_value=[body])
    return resp


class TestRTMWModelSpecVerification:
    """model_spec() must stay lazy (no network I/O) and must correctly
    surface the sidecar's filename/sha256 so OnnxSession.create() can verify
    downloads. See specs/sidecar-implementation-plan.md's M3 status note
    (same contract applies to M4).
    """

    def test_model_spec_performs_no_network_io(self):
        from skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.rtmw_wholebody_detector import (
            RTMWWholebodyDetector,
        )

        with patch(
            "skellytracker.core.sessions.model_registry.requests.get"
        ) as mock_get:
            for model_name in _MODEL_NAMES:
                RTMWWholebodyDetector.model_spec(model_name)
        mock_get.assert_not_called()

    def test_model_spec_populates_expected_filename_and_sha256(self):
        from skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.rtmw_wholebody_detector import (
            RTMWWholebodyDetector,
        )

        sidecar = load_sidecar(_RTMW_DIR / "rtmw-wholebody.yaml")

        for model_name in _MODEL_NAMES:
            spec = RTMWWholebodyDetector.model_spec(model_name)
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

        from skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.rtmw_wholebody_detector import (
            RTMWWholebodyDetector,
        )
        from skellytracker.core.sessions.model_registry import resolve_model_path

        spec = RTMWWholebodyDetector.model_spec("rtmw-l-m_256x192")
        onnx_bytes = b"fake-onnx-inside-zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("upstream_internal_name.onnx", onnx_bytes)
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
        assert path.name == spec.expected_filename
        assert path.read_bytes() == onnx_bytes

    def test_resolve_model_path_rejects_mismatched_sha256(self, tmp_path):
        from skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.rtmw_wholebody_detector import (
            RTMWWholebodyDetector,
        )
        from skellytracker.core.sessions.model_registry import (
            ModelIntegrityError,
            resolve_model_path,
        )

        spec = RTMWWholebodyDetector.model_spec("rtmw-l-m_256x192")
        body = b"fake-zip-bytes-for-rtmw"

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
