"""Exercise CPU session configuration through real ONNX inference."""
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from skellytracker.core.sessions.model_registry import ModelSource
from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
from skellytracker.core.sessions.onnx_model_spec import OnnxModelSpec


@pytest.mark.parametrize("threads, expected", [(None, 0), (0, 0), (2, 2)])
def test_cpu_thread_budget_and_inference(tmp_path, threads, expected):
    helper = onnx.helper
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "x"], ["y"])], "square",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3, 2, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "square.onnx"
    onnx.save(model, path)
    session = OnnxSession.create(OnnxSessionConfig(
        batch_size=1, execution_provider="cpu", fp16=False,
        intra_op_num_threads=threads,
        models=[OnnxModelSpec(name="square", source=ModelSource(local_path=str(path)), input_size=(2, 2))],
    ))
    try:
        runtime = session.get_session("square")
        options = runtime.get_session_options()
        assert options.intra_op_num_threads == expected
        assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
        values = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
        np.testing.assert_array_equal(runtime.run(None, {"x": values})[0], values ** 2)
    finally:
        session.close()


def test_negative_thread_budget_rejected():
    with pytest.raises(ValueError):
        OnnxSessionConfig(batch_size=1, intra_op_num_threads=-1)
