"""Real spawn/pipe lifecycle tests without GPU work or camera access."""

import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from skellytracker.core.sessions.isolated_onnx_runtime import (
    IsolatedOnnxRuntime,
    IsolatedPreparationCancelled,
    _build_native_session,
)
from skellytracker.core.sessions.model_registry import ModelSource
from skellytracker.core.sessions.onnx_model_spec import OnnxModelSpec, OnnxSessionConfig
from skellytracker.core.sessions.onnx_preparation import (
    OnnxPreparationTicket,
    PreparationPhase,
)
from skellytracker.core.sessions.onnx_session import OnnxSession


def _blocked_child(connection, config, cache_directory):
    connection.send(("phase", "blocked_fake_native"))
    time.sleep(60)


def _failed_child(connection, config, cache_directory):
    connection.send(("error", "fixture native failure"))
    connection.close()


def _crashed_child(connection, config, cache_directory):
    os._exit(7)


def _fake_ready_child(connection, config, cache_directory):
    connection.send(("ready", {
        "models": {"model": {"inputs": [("x", "tensor(float)", ("N", 3, 2, 2))],
                             "outputs": [("y", "tensor(float)", ("N", 3, 2, 2))],
                             "providers": ["CPUExecutionProvider"]}},
        "hashes": {"model": "fixture"}, "device_id": 0, "provider": "cpu",
    }))
    while True:
        command = connection.recv()
        if command[0] == "close":
            return
        connection.send(("result", [command[3]["x"] ** 2]))


def _config():
    return OnnxSessionConfig(batch_size=1, models=[], execution_provider="cpu", device_id=0)


def test_process_proxy_runs_without_moving_native_handles(tmp_path):
    runtime = IsolatedOnnxRuntime(config=_config(), cache_root=tmp_path, process_target=_fake_ready_child)
    try:
        metadata = runtime.prepare()
        preparation_elapsed = runtime.status().elapsed_seconds
        assert metadata["provider"] == "cpu"
        assert runtime.status().alive
        assert runtime.status().process_id != os.getpid()
        proxy = runtime.proxy("model")
        values = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
        np.testing.assert_array_equal(proxy.run(None, {"x": values})[0], values ** 2)
        info = proxy.get_inputs()[0]
        info.shape[0] = "changed"
        assert proxy.get_inputs()[0].shape[0] == "N"
        assert proxy.get_providers() == ["CPUExecutionProvider"]
        assert runtime.status().elapsed_seconds == preparation_elapsed
    finally:
        runtime.close()
    assert not runtime.status().alive
    assert list(tmp_path.iterdir()) == []
    runtime.close()


def test_cancel_stalled_child_confirms_exit_and_removes_only_owned_files(tmp_path):
    sentinel = tmp_path / "preserve-user-file"
    sentinel.write_text("keep")
    runtime = IsolatedOnnxRuntime(config=_config(), cache_root=tmp_path, process_target=_blocked_child)
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(runtime.prepare)
        deadline = time.monotonic() + 10
        while runtime.status().phase != "blocked_fake_native":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        runtime.request_stop()
        with pytest.raises(IsolatedPreparationCancelled):
            future.result(timeout=5)
    assert runtime.status().phase == "cancelled"
    assert not runtime.status().alive
    assert list(tmp_path.iterdir()) == [sentinel]


@pytest.mark.parametrize("target", [_failed_child, _crashed_child])
def test_failed_child_retains_error_and_releases_resources(tmp_path, target):
    runtime = IsolatedOnnxRuntime(config=_config(), cache_root=tmp_path, process_target=target)
    with pytest.raises((RuntimeError, EOFError)):
        runtime.prepare()
    assert runtime.status().phase == "failed"
    assert not runtime.status().alive
    assert list(tmp_path.iterdir()) == []


def test_stalled_preparation_deadline_terminates_owned_child(tmp_path):
    runtime = IsolatedOnnxRuntime(config=_config(), cache_root=tmp_path,
                                  preparation_timeout=0.1, process_target=_blocked_child)
    with pytest.raises(TimeoutError):
        runtime.prepare()
    assert runtime.status().phase == "failed"
    assert not runtime.status().alive
    assert list(tmp_path.iterdir()) == []


def test_native_cpu_fixture_loads_and_infers_in_child(tmp_path):
    onnx = pytest.importorskip("onnx")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Mul", ["x", "x"], ["y"])], "square",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["N", 3, 2, 2])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, ["N", 3, 2, 2])],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "fixture.onnx"
    onnx.save(model, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config = OnnxSessionConfig(batch_size=2, execution_provider="cpu", device_id=0, fp16=False,
                              models=[OnnxModelSpec(name="square", source=ModelSource(local_path=str(path)),
                                                    expected_sha256=digest, input_size=(2, 2))])
    runtime = IsolatedOnnxRuntime(config=config, cache_root=tmp_path / "jobs")
    try:
        metadata = runtime.prepare()
        assert metadata["hashes"] == {"square": digest}
        assert metadata["provider"] == "cpu"
        proxy = runtime.proxy("square")
        wrapper = OnnxSession(_sessions={"square": proxy}, _resource_owner=runtime)
        values = np.ones((2, 3, 2, 2), dtype=np.float32) * 3
        np.testing.assert_array_equal(wrapper.run("square", {"x": values})[0], values ** 2)
        wrapper.close()
        assert not runtime.status().alive
    finally:
        runtime.close()


def test_adoption_transfers_process_owner_once():
    metadata = SimpleNamespace(name="x", type="tensor(float)", shape=["N", 3, 2, 2])
    native = SimpleNamespace(get_inputs=lambda: [metadata], get_outputs=lambda: [metadata],
                             get_providers=lambda: ["TensorrtExecutionProvider"], run=lambda names, inputs: [])
    current_owner = SimpleNamespace(close=Mock(), ensure_ready=Mock())
    candidate_owner = SimpleNamespace(close=Mock(), ensure_ready=Mock())
    config = OnnxSessionConfig(batch_size=1, execution_provider="cuda", device_id=0)
    current = OnnxSession(_sessions={"model": native}, execution_provider="cuda", _creation_config=config,
                          _model_hashes={"model": "same"}, _resource_owner=current_owner)
    prepared_config = config.model_copy(update={"execution_provider": "trt"})
    candidate = OnnxSession(_sessions={"model": native}, execution_provider="trt", _creation_config=prepared_config,
                            _model_hashes={"model": "same"}, _resource_owner=candidate_owner, _prepared_warmed=True)
    retired = current.adopt_prepared_backend(candidate)
    candidate.close()
    candidate_owner.close.assert_not_called()
    retired.close()
    current_owner.close.assert_called_once()
    current.close()
    current.close()
    candidate_owner.close.assert_called_once()


def test_verified_cache_is_a_seed_but_every_worker_still_constructs_and_warms(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="fixture"))
    config = OnnxSessionConfig(batch_size=2, execution_provider="trt", device_id=0, models=[])
    seeded = []
    native = SimpleNamespace(get_providers=lambda: ["TensorrtExecutionProvider"])
    session = SimpleNamespace(_sessions={"model": native}, execution_provider="trt", close=Mock())

    def create(config, *, engine_cache_dir, progress):
        seeded.append((engine_cache_dir / "fixture.engine").exists())
        (engine_cache_dir / "fixture.engine").write_bytes(b"not a native engine")
        return session

    with patch.object(OnnxSession, "create", side_effect=create) as constructor, patch(
        "skellytracker.core.sessions.onnx_session._warmup",
    ) as warmup:
        for index, request in enumerate((config, config, config.model_copy(update={"batch_size": 3}))):
            job = tmp_path / f"job-{index}"
            job.mkdir()
            assert _build_native_session(request, job, lambda phase, model: None) is session
        assert seeded == [False, True, False]
        assert constructor.call_count == 3
        assert warmup.call_count == 4


def test_failed_warmup_does_not_publish_cache_seed(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="fixture"))
    config = OnnxSessionConfig(batch_size=2, execution_provider="trt", device_id=0, models=[])
    session = SimpleNamespace(execution_provider="trt", close=Mock())
    job = tmp_path / "job"
    job.mkdir()
    with patch.object(OnnxSession, "create", return_value=session), patch(
        "skellytracker.core.sessions.onnx_session._warmup", side_effect=RuntimeError("fixture warmup failed"),
    ), pytest.raises(RuntimeError, match="fixture warmup failed"):
        _build_native_session(config, job, lambda phase, model: None)
    session.close.assert_called_once()
    assert list((tmp_path / "completed").glob("*/preparation_complete.json")) == []


def test_close_retains_owned_handles_and_cache_until_exit_is_confirmed(tmp_path):
    runtime = IsolatedOnnxRuntime(config=_config(), cache_root=tmp_path)
    process = SimpleNamespace(is_alive=lambda: True, terminate=Mock(), kill=Mock(), join=Mock(), pid=123)
    connection = SimpleNamespace(close=Mock())
    directory = tmp_path / "trt-job-fixture"
    directory.mkdir()
    runtime._directory, runtime._owned_root = directory, tmp_path.resolve()
    runtime._process, runtime._connection = process, connection
    with pytest.raises(RuntimeError, match="still stopping"):
        runtime.close()
    assert runtime.status().alive and runtime.status().phase == "stopping"
    assert directory.exists()
    connection.close.assert_not_called()
    process.is_alive = lambda: False
    runtime.close()
    assert not directory.exists()
    connection.close.assert_called_once()


@pytest.mark.parametrize("transfer", [False, True])
def test_isolated_job_adapts_metadata_and_stops_only_its_owned_candidate(transfer):
    from skellytracker.core.sessions.isolated_onnx_preparation import (
        isolated_preparation_job,
    )
    metadata = SimpleNamespace(name="x", type="tensor(float)", shape=["N", 3, 2, 2])
    native = SimpleNamespace(get_inputs=lambda: [metadata], get_outputs=lambda: [metadata],
                             get_providers=lambda: ["TensorrtExecutionProvider"], run=lambda names, inputs: [])
    current = OnnxSession(
        _sessions={"model": native}, execution_provider="cuda", _model_hashes={"model": "known"},
        _creation_config=OnnxSessionConfig(batch_size=1, execution_provider="cuda", device_id=0),
    )
    runtime = Mock(spec=IsolatedOnnxRuntime)
    runtime.close = Mock()
    runtime.ensure_ready = Mock()
    runtime.request_stop = Mock()
    runtime.prepare.return_value = {"hashes": {"model": "known"}, "provider": "trt", "device_id": 0}
    runtime.proxy.return_value = native
    ticket = OnnxPreparationTicket("entry", 1, "known")
    with patch("skellytracker.core.sessions.isolated_onnx_preparation.IsolatedOnnxRuntime", return_value=runtime):
        job = isolated_preparation_job(current, ticket)
        job.run()
    assert job.snapshot().phase == PreparationPhase.READY
    if transfer:
        candidate = job.take_ready(expected_ticket=ticket)
        job.request_stop()
        runtime.request_stop.assert_not_called()
        retired = current.adopt_prepared_backend(candidate)
        candidate.close()
        runtime.close.assert_not_called()
        retired.close()
        current.close()
    else:
        job.request_stop()
        runtime.request_stop.assert_called_once()
        job.dispose()
    runtime.close.assert_called_once()


def test_external_tensor_model_is_rejected_before_native_construction(tmp_path):
    onnx = pytest.importorskip("onnx")
    tensor = onnx.TensorProto(name="weights", dims=[1], data_type=onnx.TensorProto.FLOAT,
                              data_location=onnx.TensorProto.EXTERNAL)
    tensor.external_data.add(key="location", value="weights.bin")
    graph = onnx.helper.make_graph([], "external", [], [], initializer=[tensor])
    path = tmp_path / "external.onnx"
    path.write_bytes(onnx.helper.make_model(graph).SerializeToString())
    config = OnnxSessionConfig(batch_size=1, execution_provider="trt", device_id=0, models=[
        OnnxModelSpec(name="external", source=ModelSource(local_path=str(path)), input_size=(2, 2),
                      expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    ])
    job = tmp_path / "job"
    job.mkdir()
    with patch.object(OnnxSession, "create") as create, pytest.raises(ValueError, match="external tensor data"):
        _build_native_session(config, job, lambda phase, model: None)
    create.assert_not_called()
