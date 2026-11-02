"""Exercise the standalone demo with a real CPU ONNX session and encoded video."""
import hashlib
from unittest.mock import Mock

import cv2
import numpy as np
import onnx
import pytest
import yaml
from onnx import TensorProto, helper, numpy_helper

from skellytracker.core.detectors.keypoint_detectors.dlc.run_demo import build_dlc_demo, main
from skellytracker.core.io.demo_manager import DemoManager
from skellytracker.core.annotation.annotator import Annotator
from skellytracker.core.tracker.tracker import Tracker
from skellytracker.core.tracker.tracker_state import TrackerState
from skellytracker.tests.test_dlc_tracker_recipe import write_sidecar


@pytest.fixture
def model(tmp_path):
    sidecar = write_sidecar(tmp_path, {"method": "fit", "target_size": [8, 12], "pad_value": 0})
    data = yaml.safe_load(sidecar.read_text())
    names = [f"feature_{i}" for i in range(50)]
    data["pose"]["tracked_points"] = names
    data["pose"]["connections"] = [{"name": "edges", "edges": [[names[49], names[0]]]}]
    data["outputs"][0].update(shape=[-1, 50, 3], keypoint_count=50)
    values = np.zeros((1, 50, 3), dtype=np.float32)
    values[0, :, 0] = np.linspace(1, 10, 50)
    values[0, :, 1] = 4
    values[0, :, 2] = 0.9
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["keypoints"], value=numpy_helper.from_array(values))],
        "demo_fixture", [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 8, 12])],
        [helper.make_tensor_value_info("keypoints", TensorProto.FLOAT, [1, 50, 3])],
    )
    artifact = sidecar.parent / "pose_model.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), artifact)
    data["sizes"]["native"]["onnx"]["batch_artifacts"]["dynamic"]["precision_artifacts"]["fp32"]["url_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    sidecar.write_text(yaml.safe_dump(data))
    return sidecar, tuple(names)


def test_demo_preserves_fifty_point_identity_and_declared_connections(model):
    sidecar, names = model
    demo = build_dlc_demo(sidecar)
    try:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        observation, _ = demo.tracker.process_image(frame, 0, TrackerState(), 0)
        stage = observation.stages["dlc_keypoints"]
        assert tuple(stage.keypoints.names) == names
        assert stage.keypoints.xyz.shape == (50, 3)
        schema = demo.annotator.config.stage_schemas["dlc_keypoints"]
        assert schema.connections == ((names[49], names[0]),)
        annotated = demo.annotator.annotate(frame, observation)
        assert annotated.any() and not frame.any()
    finally:
        demo.tracker.close()


def test_headless_video_uses_real_tracker_and_writes_all_frames(model, tmp_path, monkeypatch, capsys):
    sidecar, _ = model
    source, output = tmp_path / "source.avi", tmp_path / "annotated.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    assert writer.isOpened()
    for index in range(4):
        writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
    writer.release()
    for name in ("namedWindow", "imshow", "waitKey", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, Mock(side_effect=AssertionError("Headless must not access GUI")))
    main(["--sidecar", str(sidecar), "--video", str(source), "--headless", "--output-video", str(output)])
    assert "Processed 4 frames" in capsys.readouterr().out
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
        ok, image = capture.read()
        assert ok and image.shape == (48, 64, 3) and image.any()
    finally:
        capture.release()


def test_video_open_failure_closes_tracker(tmp_path):
    tracker = Mock(spec=Tracker)
    demo = DemoManager(tracker=tracker, annotator=Mock(spec=Annotator))
    with pytest.raises(RuntimeError, match="Could not open video"):
        demo.run_video(tmp_path / "missing.mp4", headless=True)
    tracker.close.assert_called_once()


@pytest.mark.parametrize("args", [
    ["--headless"], ["--confidence", "1.1"], ["--provider", "trt"],
    ["--video", "same.mp4", "--output-video", "same.mp4"],
])
def test_cli_rejects_invalid_options_before_model_loading(args):
    with pytest.raises(SystemExit) as exc:
        main(["--sidecar", "not-loaded.yaml", *args])
    assert exc.value.code == 2
