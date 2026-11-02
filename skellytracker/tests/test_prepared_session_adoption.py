"""CPU-only native facades: test ownership, not GPU resource-transfer safety."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import onnxruntime as ort
import pytest

from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.sessions.onnx_model_spec import OnnxSessionConfig
from skellytracker.core.sessions.onnx_preparation import (
    OnnxPreparationJob,
    OnnxPreparationTicket,
)
from skellytracker.core.sessions.onnx_session import OnnxSession
from skellytracker.core.sessions.shared_sessions import (
    SessionRequest,
    SharedSessionPool,
)


def session(provider="cuda"):
    native = Mock(spec=ort.InferenceSession)
    native.get_inputs.return_value = [SimpleNamespace(name="input", type="tensor(float)", shape=["N", 3, 8, 8])]
    native.get_outputs.return_value = [SimpleNamespace(name="output", type="tensor(float)", shape=["N", 17, 3])]
    native.get_providers.return_value = ["TensorrtExecutionProvider" if provider == "trt" else "CUDAExecutionProvider"]
    return OnnxSession(
        _sessions={"pose": native}, execution_provider=provider, device_id=0, batch_size=2,
        _creation_config=OnnxSessionConfig(batch_size=2, execution_provider=provider, device_id=0),
        _model_hashes={"pose": "known-model-hash"}, _prepared_warmed=provider == "trt",
    )


def test_native_handoff_preserves_wrapper_and_defers_cleanup():
    current, candidate = session(), session("trt")
    wrapper = current
    old_native = current.get_session("pose")
    new_native = candidate.get_session("pose")
    with patch("skellytracker.core.sessions.onnx_session.gc.collect") as collect:
        retired = current.adopt_prepared_backend(candidate)
        collect.assert_not_called()
        assert current is wrapper
        assert current.get_session("pose") is new_native
        assert retired.get_session("pose") is old_native
        assert candidate._sessions == {}
        retired.close()
        collect.assert_called_once()


@pytest.mark.parametrize("change", ["device", "capacity", "hash", "precision", "provider", "contract", "warmup"])
def test_incompatible_candidate_leaves_both_resources_owned(change):
    current, candidate = session(), session("trt")
    old_native = current.get_session("pose")
    if change == "device":
        candidate.device_id = 1
    elif change == "capacity":
        candidate.batch_size = 3
    elif change == "hash":
        candidate._model_hashes["pose"] = "wrong"
    elif change == "precision":
        candidate._creation_config.fp16 = False
    elif change == "provider":
        candidate.get_session("pose").get_providers.return_value = ["CUDAExecutionProvider"]
    elif change == "contract":
        candidate.get_session("pose").get_outputs.return_value[0].shape = ["N", 18, 3]
    else:
        candidate._prepared_warmed = False
    with pytest.raises(ValueError):
        current.adopt_prepared_backend(candidate)
    assert current.get_session("pose") is old_native
    assert candidate._sessions


def request(provider=None):
    return SessionRequest(session_type=OnnxSession, config=OnnxSessionConfig(batch_size=2, execution_provider=provider))


def test_pool_isolates_pinned_cuda_and_coalesces_automatic_clients():
    pool = SharedSessionPool()
    with patch.object(OnnxSession, "create", side_effect=lambda config: session(config.execution_provider)):
        first = pool.create_tracker(config=TrackerConfig(stages=[]), requests=(request(),), cuda_first=True)
        second = pool.create_tracker(config=TrackerConfig(stages=[]), requests=(request(),), cuda_first=True)
        pinned = pool.create_tracker(config=TrackerConfig(stages=[]), requests=(request("cuda"),), cuda_first=True)
    assert first._entries[0] is second._entries[0]
    assert first._entries[0] is not pinned._entries[0]
    candidate = session("trt")
    job = OnnxPreparationJob(ticket=OnnxPreparationTicket("entry", 0, "identity"),
                             build=lambda: candidate, validate=lambda _: None)
    with patch.object(OnnxSession, "preparation_job", return_value=job) as prepare:
        assert pool.prepare_onnx(lease=first) is job
        assert pool.prepare_onnx(lease=second) is job
        assert pool.prepare_onnx(lease=pinned) is None
        prepare.assert_called_once()
    job.run()
    tracker = first.tracker
    with ThreadPoolExecutor(max_workers=1) as executor, pytest.raises(RuntimeError, match="owning inference thread"):
        executor.submit(pool.activate_prepared, lease=first, job=job).result()
    result = pool.activate_prepared(lease=first, job=job)
    assert result.activated
    assert first.tracker is tracker
    assert second._entries[0].session.execution_provider == "trt"
    assert pinned._entries[0].session.execution_provider == "cuda"
    result.cleanup.close()
    repeated = pool.activate_prepared(lease=second, job=job)
    assert repeated.activated and repeated.cleanup is None
    first.close()
    with pytest.raises(ValueError, match="no longer owned"):
        pool.activate_prepared(lease=first, job=job)
    pool.close()


def test_rejected_adoption_returns_candidate_for_worker_cleanup():
    pool = SharedSessionPool()
    with patch.object(OnnxSession, "create", return_value=session()):
        lease = pool.create_tracker(config=TrackerConfig(stages=[]), requests=(request(),), cuda_first=True)
    candidate = session("trt")
    candidate.device_id = 1
    job = OnnxPreparationJob(ticket=OnnxPreparationTicket("entry", 0, "identity"),
                             build=lambda: candidate, validate=lambda _: None)
    lease._entries[0].preparation = job
    job.run()
    result = pool.activate_prepared(lease=lease, job=job)
    assert not result.activated and result.error
    assert result.cleanup is candidate and candidate._sessions
    assert lease._entries[0].session.execution_provider == "cuda"
    result.cleanup.close()
    pool.close()


def test_unsupported_recipe_cannot_share_an_upgradeable_entry():
    pool = SharedSessionPool()
    def validator(config):
        if config.execution_provider == "trt":
            raise ValueError("TRT unsupported")
    with patch.object(OnnxSession, "create", return_value=session()), patch(
        "skellytracker.core.tracker.tracker_factory.collect_required_sessions",
        return_value=SimpleNamespace(onnx_validators=[validator]),
    ):
        lease = pool.create_tracker(config=TrackerConfig(stages=[]), requests=(request(),), cuda_first=True)
    assert not lease._entries[0].upgradeable
    assert pool.prepare_onnx(lease=lease) is None
    pool.close()
