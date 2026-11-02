import hashlib
from unittest.mock import Mock, patch

import onnxruntime as ort
import pytest

from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.detectors.keypoint_detectors.dlc import (
    DlcDetector,
    DlcDetectorConfig,
)
from skellytracker.core.sessions.model_registry import ModelSource, resolve_model_path
from skellytracker.core.sessions.onnx_model_spec import OnnxModelSpec, OnnxSessionConfig
from skellytracker.core.sessions.onnx_session import OnnxSession
from skellytracker.core.sessions.ort_session_utils import cuda_provider_options
from skellytracker.core.sessions.shared_sessions import (
    SessionRequest,
    SharedSessionPool,
)
from skellytracker.core.sidecar.model import PrecisionArtifact
from skellytracker.core.tracker.tracker_factory import tracker_session_requests


def model_spec(tmp_path, precision=False):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"verified fixture")
    return OnnxModelSpec(
        name="fixture",
        source=ModelSource(local_path=str(path)),
        input_size=(8, 8),
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cuda_use_tf32=precision,
    )


@pytest.mark.parametrize(
    "provider,fp16",
    [(None, False), ("trt", False), ("coreml", False), ("cpu", True), ("cuda", True)],
)
def test_dlc_rejects_uncertified_session_modes(provider, fp16):
    with pytest.raises(ValueError, match="explicit CPU or CUDA"):
        DlcDetector.validate_session_config(
            OnnxSessionConfig(batch_size=1, execution_provider=provider, fp16=fp16)
        )


@pytest.mark.parametrize("provider", ["cpu", "cuda"])
def test_dlc_accepts_explicit_fp32(provider):
    DlcDetector.validate_session_config(
        OnnxSessionConfig(batch_size=1, execution_provider=provider, fp16=False)
    )


def test_local_model_hash_checked_before_runtime_allocation(tmp_path):
    spec = model_spec(tmp_path)
    assert resolve_model_path(
        spec.source, expected_sha256=spec.expected_sha256
    ).exists()
    assert resolve_model_path(spec.source).exists()
    spec.expected_sha256 = "0" * 64
    with (
        patch("skellytracker.core.sessions.onnx_session._verify_ort_install"),
        patch("skellytracker.core.sessions.onnx_session.require_provider"),
        patch(
            "skellytracker.core.sessions.onnx_session.build_tuned_ort_session"
        ) as allocate,
    ):
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            OnnxSession.create(
                OnnxSessionConfig(
                    batch_size=1, models=[spec], execution_provider="cpu", fp16=False
                )
            )
        allocate.assert_not_called()


def test_creation_config_is_defensively_owned(tmp_path):
    config = OnnxSessionConfig(
        batch_size=1,
        models=[model_spec(tmp_path)],
        execution_provider="cpu",
        fp16=False,
    )
    with (
        patch("skellytracker.core.sessions.onnx_session._verify_ort_install"),
        patch("skellytracker.core.sessions.onnx_session.require_provider"),
        patch(
            "skellytracker.core.sessions.onnx_session.build_tuned_ort_session",
            return_value=Mock(spec=ort.InferenceSession),
        ),
        patch("skellytracker.core.sessions.onnx_session._warmup"),
    ):
        session = OnnxSession.create(config)
    config.models[0].cuda_use_tf32 = True
    assert session.creation_config.models[0].cuda_use_tf32 is False
    snapshot = session.creation_config
    snapshot.models.clear()
    assert len(session.creation_config.models) == 1
    session.close()


def test_precision_options_separate_pool_sessions(tmp_path):
    pool = SharedSessionPool()
    leases = []
    try:
        with patch.object(
            OnnxSession, "create", side_effect=lambda config: OnnxSession()
        ):
            for precision in (None, False, True, False):
                config = OnnxSessionConfig(
                    batch_size=1, models=[model_spec(tmp_path, precision)]
                )
                request = SessionRequest(session_type=OnnxSession, config=config)
                leases.append(
                    pool.create_tracker(
                        config=TrackerConfig(stages=[]), requests=(request,)
                    )
                )
        sessions = [lease._entries[0].session for lease in leases]
        assert len({id(session) for session in sessions[:3]}) == 3
        assert sessions[1] is sessions[3]
    finally:
        for lease in leases:
            lease.close()
        pool.close()
    assert "use_tf32" not in cuda_provider_options()
    assert cuda_provider_options(use_tf32=False)["use_tf32"] == "0"
    assert cuda_provider_options(use_tf32=True)["use_tf32"] == "1"


@pytest.mark.parametrize("precision", [None, False, True])
def test_precision_sidecar_roundtrip(precision):
    artifact = PrecisionArtifact(filename="pose.onnx", cuda_use_tf32=precision)
    restored = PrecisionArtifact.model_validate_json(artifact.model_dump_json())
    assert restored.cuda_use_tf32 is precision


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_provenance",
        "wrong_descriptor",
        "wrong_provider",
        "cpu_fallback",
        "tf32",
    ],
)
def test_supplied_session_rejects_unverifiable_or_uncertified_model(tmp_path, fault):
    spec = model_spec(tmp_path)
    config = DlcDetectorConfig(model_name=spec.name)
    spec.name = config.model_name
    creation = OnnxSessionConfig(
        batch_size=1, models=[spec], execution_provider="cuda", fp16=False
    )
    runtime = Mock(spec=ort.InferenceSession)
    runtime.get_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    runtime.get_provider_options.return_value = {
        "CUDAExecutionProvider": {"use_tf32": "0"}
    }
    session = OnnxSession(
        _sessions={spec.name: runtime},
        execution_provider="cuda",
        _creation_config=creation.model_copy(deep=True),
    )
    if fault == "unknown_provenance":
        session._creation_config = None
    elif fault == "wrong_descriptor":
        session._creation_config.models[0].expected_sha256 = "0" * 64
    elif fault == "wrong_provider":
        session.execution_provider = "cpu"
    elif fault == "cpu_fallback":
        runtime.get_providers.return_value = ["CPUExecutionProvider"]
    else:
        runtime.get_provider_options.return_value = {
            "CUDAExecutionProvider": {"use_tf32": "1"}
        }
    with (
        patch.object(DlcDetector, "model_spec", return_value=spec),
        pytest.raises(ValueError),
    ):
        DlcDetector.create(config, session)
    runtime.run.assert_not_called()
    session.close()


def test_factory_validates_dlc_before_allocation_and_preserves_other_defaults(tmp_path):
    from skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.rtmw_wholebody_detector import (
        RTMWWholebodyDetectorConfig,
    )

    spec = model_spec(tmp_path)
    dlc = DlcDetectorConfig(model_name=spec.name)
    rtmw = RTMWWholebodyDetectorConfig()
    ordinary = TrackerConfig(
        stages=[DetectionStageConfig(name="body", keypoint_detectors=[rtmw])]
    )
    ordinary_request = tracker_session_requests(ordinary)[0]
    assert ordinary_request.config.execution_provider is None
    assert ordinary_request.config.fp16 is True
    mixed = TrackerConfig(
        stages=[DetectionStageConfig(name="combined", keypoint_detectors=[rtmw, dlc])]
    )
    with (
        patch.object(DlcDetector, "model_spec", return_value=spec),
        patch.object(OnnxSession, "create") as allocate,
    ):
        with pytest.raises(ValueError, match="explicit CPU or CUDA"):
            tracker_session_requests(mixed)
        allocate.assert_not_called()
        request = tracker_session_requests(
            mixed, onnx_overrides={"execution_provider": "cpu", "fp16": False}
        )[0]
        assert len(request.config.models) == 2
