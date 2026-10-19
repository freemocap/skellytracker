"""The webcam entry points configure one shared demo without opening a camera."""
import sys
from unittest.mock import Mock

import pytest

from skellytracker.core.detectors.keypoint_detectors.rtmpose import run_demo
from skellytracker.core.detectors.keypoint_detectors.rtmpose.wholebody import run_demo as wholebody_demo


@pytest.mark.parametrize("entrypoint", [run_demo.main, wholebody_demo.main])
@pytest.mark.parametrize("arguments, interval", [([], 100), (["--redetect-frames", "1"], 1)])
def test_demo_cli_passes_cadence_and_camera(monkeypatch, entrypoint, arguments, interval):
    builder = Mock()
    monkeypatch.setattr(run_demo, "build_rtmpose_demo", builder)
    monkeypatch.setattr(sys, "argv", ["demo", "--provider", "cpu", "--camera", "2", *arguments])
    entrypoint()
    assert builder.call_args.kwargs["redetect_frames"] == interval
    assert builder.call_args.kwargs["provider"] == "cpu"
    builder.return_value.run_webcam.assert_called_once_with(camera_index=2)


@pytest.mark.parametrize("interval", [0, -1])
def test_invalid_interval_rejected_before_model_loading(monkeypatch, interval):
    builder = Mock()
    monkeypatch.setattr(run_demo, "build_rtmpose_demo", builder)
    monkeypatch.setattr(sys, "argv", ["demo", "--redetect-frames", str(interval)])
    with pytest.raises(SystemExit) as exc:
        run_demo.main()
    assert exc.value.code == 2
    builder.assert_not_called()


def test_no_detector_cli_remains_available(monkeypatch):
    builder = Mock()
    monkeypatch.setattr(run_demo, "build_rtmpose_demo", builder)
    monkeypatch.setattr(sys, "argv", ["demo", "--no-detector", "--provider", "cpu"])
    run_demo.main()
    assert builder.call_args.kwargs["use_person_detector"] is False
