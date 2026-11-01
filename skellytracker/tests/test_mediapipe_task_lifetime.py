import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.sessions.mediapipe_session import MediaPipeSession
from skellytracker.core.tracker.detection_stage import DetectionStage
from skellytracker.core.tracker.tracker_state import StageState


@pytest.mark.parametrize("folder,name", [("hands", "Hand"), ("body", "Pose"), ("face", "Face")])
def test_native_tasks_created_only_for_used_camera_streams(folder, name, monkeypatch):
    from mediapipe.tasks.python import vision
    module = importlib.import_module(
        f"skellytracker.core.detectors.keypoint_detectors.mediapipe.{folder}.mediapipe_{name.lower()}_detector")
    native_tasks = []

    def create_task(options):
        task = Mock()
        task.detect_for_video.return_value = SimpleNamespace(
            pose_landmarks=[], face_landmarks=[], hand_landmarks=[], handedness=[])
        native_tasks.append(task)
        return task

    monkeypatch.setattr(getattr(vision, f"{name}Landmarker"), "create_from_options", create_task)
    config = DetectionStageConfig(name="test_stage", keypoint_detectors=[
        getattr(module, f"Mediapipe{name}DetectorConfig")()])
    session = MediaPipeSession.create()
    stage = DetectionStage.create(config, {"mediapipe": session})
    try:
        assert not native_tasks
        images = {f"camera_{index}": np.zeros((80, 80, 3), np.uint8) for index in range(3)}
        _, states = stage.run_batch(images, {}, DetectionContext(timestamp_ms=0))
        assert len(native_tasks) == 3
        stage.run_batch(images, states, DetectionContext(frame_number=1, timestamp_ms=33))
        assert len(native_tasks) == 3
        assert all(task.detect_for_video.call_count == 2 for task in native_tasks)
        stage.reset_temporal_state()
        assert len(native_tasks) == 3  # Reset must not recreate unused native tasks.
        assert all(task.close.call_count == 1 for task in native_tasks)
        stage.run_batch(images, {}, DetectionContext(timestamp_ms=0))
        assert len(native_tasks) == 6
        stage.run(images["camera_0"], StageState(), context=DetectionContext(timestamp_ms=0))
        assert len(native_tasks) == 7  # Separate single-image stream, on demand.
    finally:
        stage.close()
        stage.close()
        session.close()
    assert all(task.close.call_count == 1 for task in native_tasks)
