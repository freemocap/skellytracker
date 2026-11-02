"""Prepared builder tests replace every native construction and inference call."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import onnxruntime as ort
import pytest

from skellytracker.core.sessions.model_registry import ModelSource
from skellytracker.core.sessions.onnx_model_spec import OnnxModelSpec, OnnxSessionConfig
from skellytracker.core.sessions.onnx_preparation import (
    OnnxPreparationTicket,
    PreparationPhase,
)
from skellytracker.core.sessions.onnx_prepared_backend import preparation_job
from skellytracker.core.sessions.onnx_session import OnnxSession


@pytest.mark.parametrize("warmup_failure", [False, True])
def test_builder_pins_resolved_bytes_isolates_cache_and_warms_crop_range(tmp_path, monkeypatch, warmup_failure):
    path = tmp_path / "resolved.onnx"
    path.write_bytes(b"fake model; native constructor is mocked")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    native = Mock(spec=ort.InferenceSession)
    native.get_inputs.return_value = [SimpleNamespace(name="input", type="tensor(float)", shape=["N", 3, 8, 8])]
    native.get_outputs.return_value = [SimpleNamespace(name="output", type="tensor(float)", shape=["N", 17, 3])]
    native.get_providers.return_value = ["TensorrtExecutionProvider"]
    config = OnnxSessionConfig(batch_size=3, execution_provider="cuda", models=[
        OnnxModelSpec(name="pose", source=ModelSource(url="https://invalid.example/model.onnx"), input_size=(8, 8)),
    ])
    current = OnnxSession(_sessions={"pose": native}, execution_provider="cuda", device_id=2, batch_size=3,
                          _creation_config=config, _model_paths={"pose": path}, _model_hashes={"pose": digest})
    candidate = OnnxSession(_sessions={"pose": native}, execution_provider="trt", device_id=2, batch_size=3,
                            _model_hashes={"pose": digest})
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="test-runtime"))
    ticket = OnnxPreparationTicket("entry", 0, "resolved")
    job = preparation_job(current, ticket)
    assert not (tmp_path / ".cache").exists()
    warmed = []

    def warmup(_, specs, provider, *, batch_size):
        warmed.append(batch_size)
        if warmup_failure:
            raise RuntimeError("warmup failed")

    with patch.object(OnnxSession, "create", return_value=candidate) as create, patch(
        "skellytracker.core.sessions.onnx_prepared_backend._warmup", side_effect=warmup,
    ), patch.object(OnnxSession, "close") as close:
        if warmup_failure:
            with pytest.raises(RuntimeError, match="warmup failed"):
                job.run()
            close.assert_called_once()
            assert job.snapshot().phase == PreparationPhase.FAILED
        else:
            job.run()
            assert warmed == [1, 2]
            assert job.take_ready(expected_ticket=ticket) is candidate
            assert candidate._prepared_warmed
            close.assert_not_called()
        actual = create.call_args.args[0]
        assert actual.execution_provider == "trt" and actual.device_id == 2
        assert actual.models[0].source.local_path == str(path)
        assert actual.models[0].expected_sha256 == digest
        assert actual.models[0].prepare is None
        assert create.call_args.kwargs["engine_cache_dir"].is_relative_to(tmp_path / ".cache" / "skellytracker" / "trt_prepared")
    assert current.execution_provider == "cuda"
