"""Session construction must prepare partial batches before live inference."""

from types import SimpleNamespace
import os
from pathlib import Path
import time

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from skellytracker.core.sessions import onnx_session, ort_session_utils
from skellytracker.core.sessions.model_registry import ModelSource
from skellytracker.core.sessions.onnx_model_spec import OnnxModelSpec, OnnxSessionConfig


@pytest.mark.parametrize("provider", ["trt", "cuda", "cpu"])
@pytest.mark.parametrize("batch_dimension", ["batch", 4])
def test_session_prepares_partial_batch_range(tmp_path, monkeypatch, provider, batch_dimension):
    path = tmp_path / "model.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output"])], "test",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [batch_dimension, 3, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_dimension, 3, 8, 8])],
    )
    onnx.save(helper.make_model(graph), path)
    captured = {}

    class Runtime(ort_session_utils.ort.InferenceSession):
        def __init__(self, path_or_bytes, sess_options, providers):
            captured.update(dict(p for p in providers if isinstance(p, tuple)))
            self._providers = [p[0] if isinstance(p, tuple) else p for p in providers]

        def get_inputs(self):
            return [SimpleNamespace(name="images")]

        def run(self, names, inputs):
            return [inputs["images"].copy()]

    monkeypatch.setattr(ort_session_utils.ort, "InferenceSession", Runtime)
    monkeypatch.setattr(ort_session_utils, "_default_engine_cache_dir", lambda: tmp_path / "engines")
    monkeypatch.setattr(ort_session_utils, "validate_engine_cache", lambda path: None)
    monkeypatch.setattr(onnx_session, "require_provider", lambda provider: None)
    monkeypatch.setattr(onnx_session, "_load_nvidia_dlls_on_windows", lambda: None)
    monkeypatch.setattr(onnx_session.ort, "preload_dlls", lambda: None)
    session = onnx_session.OnnxSession.create(OnnxSessionConfig(
        batch_size=4, execution_provider=provider, device_id=0, gpu_mem_limit=1024,
        models=[OnnxModelSpec(name="test", source=ModelSource(local_path=str(path)), input_size=(8, 8))],
    ))
    try:
        if provider == "trt" and batch_dimension == "batch":
            options = captured["TensorrtExecutionProvider"]
            assert options.get("trt_profile_min_shapes") == "images:1x3x8x8"
            assert options.get("trt_profile_opt_shapes") == "images:4x3x8x8"
            assert options.get("trt_profile_max_shapes") == "images:4x3x8x8"
            assert Path(options["trt_engine_cache_path"]).name == "batch_4"
        elif provider == "trt":
            assert "trt_profile_min_shapes" not in captured["TensorrtExecutionProvider"]
        else:
            assert "TensorrtExecutionProvider" not in captured
        for count in ([4, 1, 3, 2, 4] if batch_dimension == "batch" else [4]):
            tensors = {str(i): np.full((3, 8, 8), i, np.float32) for i in range(count)}
            result = session.run_batched("test", tensors)
            assert list(result) == list(tensors)
            for key, tensor in tensors.items():
                np.testing.assert_array_equal(result[key][0], tensor[None])
    finally:
        session.close()


@pytest.mark.skipif(not os.environ.get("SKELLYTRACKER_TRT_MODEL_DIR"),
                    reason="Opt-in TensorRT GPU test requires local model directory")
@pytest.mark.parametrize("filename,size", [
    ("rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx", (256, 192)),
    ("yolox_m_8xb8-300e_humanart-c2c7a14a.onnx.prenms.onnx", (640, 640)),
])
def test_tensorrt_visibility_changes_do_not_rebuild_engines(filename, size):
    """Real GPU regression: startup may compile, changing visibility must not.

    Set SKELLYTRACKER_TRT_MODEL_DIR to an existing model cache to enable.
    Uses the normal TensorRT cache; no downloads or camera hardware required.
    """
    source = Path(os.environ["SKELLYTRACKER_TRT_MODEL_DIR"]) / filename
    session = onnx_session.OnnxSession.create(OnnxSessionConfig(
        batch_size=4, execution_provider="trt", device_id=0,
        models=[OnnxModelSpec(name="test", source=ModelSource(local_path=str(source)), input_size=size)],
    ))
    try:
        runtime = session.get_session("test")
        assert "TensorrtExecutionProvider" in runtime.get_providers()
        cache = Path(runtime.get_provider_options()["TensorrtExecutionProvider"]["trt_engine_cache_path"])
        before = {p.name: p.stat().st_mtime_ns for p in cache.glob("*.engine")}
        assert before
        for count in [4, 1, 3, 2, 4, 1, 4]:
            tensors = {str(i): np.zeros((3, *size), np.float32) for i in range(count)}
            start = time.perf_counter()
            outputs = session.run_batched("test", tensors)
            elapsed = time.perf_counter() - start
            assert elapsed < 2, f"Batch {count} stalled for {elapsed:.2f}s"
            assert list(outputs) == list(tensors)
            assert all(np.isfinite(array).all() for values in outputs.values() for array in values)
        after = {p.name: p.stat().st_mtime_ns for p in cache.glob("*.engine")}
        assert after == before, "Visibility changes must not rebuild cached engines"
    finally:
        session.close()
