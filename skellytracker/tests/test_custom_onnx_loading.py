"""Execute local synthetic models; this does not certify a DLC export or decoder."""

from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest
import yaml

from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
from skellytracker.core.sessions.session_errors import SessionCreationError
from skellytracker.core.sidecar.loader import load_sidecar
from skellytracker.core.sidecar.runtime import sidecar_model_spec
from skellytracker.core.sidecar.model import PoseDecodeSpec
from skellytracker.core.detectors.processing.simcc_decode import simcc_pose_decode


def local_coordinate_fixture(tmp_path, layout="NCHW"):
    """Return channel means as one synthetic keypoint; no trained inference claim."""
    shape = ["batch", 3, 8, 10] if layout == "NCHW" else ["batch", 8, 10, 3]
    axes = [2, 3] if layout == "NCHW" else [1, 2]
    graph = helper.make_graph(
        [
            helper.make_node(
                "ReduceMean", ["images"], ["means"], axes=axes, keepdims=0
            ),
            helper.make_node("Unsqueeze", ["means", "output_axes"], ["keypoints"]),
        ],
        "local_fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)],
        [
            helper.make_tensor_value_info(
                "keypoints", TensorProto.FLOAT, ["batch", 1, 3]
            )
        ],
        [helper.make_tensor("output_axes", TensorProto.INT64, [1], [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    model_path = tmp_path / "coordinate_fixture.onnx"
    onnx.save(model, model_path)
    data = dict(
        schema_version="v2024.09.1019",
        model_id="coordinate-fixture",
        display_name="Synthetic coordinate fixture",
        role=["pose_estimator"],
        input=dict(
            name="images",
            dtype={"fp32": "float32"},
            layout=layout,
            normalization="none",
            resize=dict(method="affine_person_crop"),
        ),
        outputs=[
            dict(
                name="keypoints",
                dtype={"fp32": "float32"},
                semantic="keypoints",
                keypoint_axis=1,
                keypoint_count=1,
            )
        ],
        pose=dict(
            estimator_type="top_down_single_person",
            requires_object_detector=True,
            landmark_schema="synthetic_fixture",
            tracked_points=["test_keypoint"],
            connections=[],
            decode=dict(method="coordinate", coordinate_scale="pixel"),
        ),
        sizes={
            "test": dict(
                input=dict(shape=[-1, *shape[1:]], resize=dict(target_size=[8, 10])),
                onnx=dict(
                    batch_artifacts={
                        str(batch): dict(
                            output_shapes=[[batch, 1, 3]],
                            precision_artifacts={
                                "fp32": dict(filename=model_path.name)
                            },
                        )
                        for batch in (1, 3)
                    }
                ),
            )
        },
    )
    sidecar_path = tmp_path / "coordinate-fixture.yaml"
    sidecar_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_sidecar(sidecar_path, cache_dir=tmp_path)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_local_sidecar_model_loads_and_executes_without_download(tmp_path, batch_size):
    sidecar = local_coordinate_fixture(tmp_path)
    spec = sidecar_model_spec(
        sidecar,
        size="test",
        batch_key=str(batch_size),
        precision="fp32",
        name="custom_coordinates",
        sidecar_dir=tmp_path,
    )
    with patch(
        "skellytracker.core.sessions.model_registry.requests.get",
        side_effect=AssertionError("Unexpected network access"),
    ):
        session = OnnxSession.create(
            OnnxSessionConfig(
                batch_size=batch_size, models=[spec], execution_provider="cpu"
            )
        )
        try:
            values = np.arange(batch_size * 3 * 8 * 10, dtype=np.float32).reshape(
                batch_size, 3, 8, 10
            )
            (output,) = session.run("custom_coordinates", {"images": values})
            np.testing.assert_allclose(
                output[:, 0], values.mean(axis=(2, 3)), rtol=1e-6
            )
            assert output.shape == (batch_size, 1, 3)
        finally:
            session.close()
    assert not session._sessions


def test_nhwc_descriptor_does_not_imply_session_warmup_support(tmp_path):
    sidecar = local_coordinate_fixture(tmp_path, layout="NHWC")
    spec = sidecar_model_spec(
        sidecar,
        size="test",
        batch_key="1",
        precision="fp32",
        name="custom_nhwc",
        sidecar_dir=tmp_path,
    )
    assert spec.input_size == (8, 10)
    with pytest.raises(SessionCreationError, match="Warmup inference failed"):
        OnnxSession.create(
            OnnxSessionConfig(batch_size=1, models=[spec], execution_provider="cpu")
        )


@pytest.mark.parametrize("method", ["coordinate", "heatmap"])
def test_descriptor_does_not_supply_an_unimplemented_pose_decoder(method):
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        simcc_pose_decode(
            (), np.zeros(2), np.ones(2), (10, 8), PoseDecodeSpec(method=method)
        )
