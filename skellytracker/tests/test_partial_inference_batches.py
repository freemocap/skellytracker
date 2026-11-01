"""Exercise partial batches through real stage/session code without model downloads."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper
import pytest

from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.detectors.detector_base_classes import KeypointDetector, ObjectDetector
from skellytracker.core.sessions.onnx_session import OnnxSession
from skellytracker.core.sessions.session_errors import InferencePipelineError
from skellytracker.core.tracker.detection_stage import DetectionStage


def make_session(batch_dimension="batch"):
    shape = [batch_dimension, 3, 8, 8]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output"])],
        "batch_identity",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    runtime = ort.InferenceSession(model.SerializeToString(), options,
                                  providers=["CPUExecutionProvider"])
    return OnnxSession(_sessions={"identity": runtime}, batch_size=4)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_supported_batch_sizes_are_quiet_and_preserve_identity(count, caplog):
    session = make_session()
    tensors = {f"camera-{i}": np.full((3, 8, 8), i, np.float32)
               for i in reversed(range(count))}
    try:
        with caplog.at_level(logging.WARNING):
            result = session.run_batched("identity", tensors)
        assert list(result) == list(tensors)
        for camera, tensor in tensors.items():
            np.testing.assert_array_equal(result[camera][0], tensor[None])
        assert not caplog.records
    finally:
        session.close()


def test_unsupported_static_batch_still_raises_model_error():
    session = make_session(batch_dimension=4)
    try:
        with pytest.raises(InferencePipelineError, match="identity"):
            session.run_batched("identity", {"camera": np.ones((3, 8, 8), np.float32)})
    finally:
        session.close()


@pytest.mark.parametrize("present", [(), ("west",), ("west", "south"),
                                     ("west", "south", "north", "east")])
def test_stage_skips_missing_crops_but_retains_every_camera(present, caplog):
    session = make_session()
    session.run = Mock(wraps=session.run)
    preprocess = Mock(side_effect=lambda image: (image.transpose(2, 0, 1).astype(np.float32), None))
    detector = Mock(spec=KeypointDetector,
        session=session, config=SimpleNamespace(model_name="identity"),
        point_names=("test_point",), preprocess=preprocess,
        postprocess=lambda raw, meta: Keypoints(
            names=("test_point",), xyz=np.full((1, 3), raw[0][0, 0, 0, 0], dtype=np.float64),
            visibility=np.ones(1)),
        close=lambda: None,
    )
    objects = Mock(spec=ObjectDetector,
        session=SimpleNamespace(kind="cpu"), close=lambda: None,
        detect=lambda image, *args: [BoundingBox.full_image(8, 8)] if image[0, 0, 0] else [],
    )
    stage = DetectionStage(name="person", keypoint_detectors=[detector], object_detector=objects)
    cameras = ("west", "north", "south", "east")
    images = {camera: np.full((8, 8, 3), i + 1 if camera in present else 0, np.uint8)
              for i, camera in enumerate(cameras)}
    try:
        with caplog.at_level(logging.WARNING):
            observations, states = stage.run_batch(images, {})
        assert list(observations) == list(cameras)
        assert set(states) == set(cameras)
        assert preprocess.call_count == len(present)
        assert session.run.call_count == bool(present)
        if present:
            submitted = session.run.call_args.args[1]["images"]
            assert submitted.shape == (len(present), 3, 8, 8)
            assert np.all(submitted > 0)  # No placeholder images were submitted.
        for i, camera in enumerate(cameras):
            points = observations[camera].keypoints
            assert points.names == ("test_point",)
            if camera in present:
                np.testing.assert_array_equal(points.xyz, np.full((1, 3), i + 1))
                np.testing.assert_array_equal(points.visibility, np.ones(1))
            else:
                assert np.isnan(points.xyz).all()
                np.testing.assert_array_equal(points.visibility, np.zeros(1))
        assert not caplog.records
    finally:
        stage.close()
        session.close()
