from __future__ import annotations

import numpy as np
import pytest

import skellytracker.core.detectors.keypoint_detectors.mediapipe  # noqa: F401 — triggers registry side-effects
from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.config.detector_configs import ObjectDetectorConfig
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.detectors.detector_base_classes import OBJECT_DETECTOR_REGISTRY
from skellytracker.core.detectors.keypoint_detectors.aruco.aruco_detector_config import (
    ArucoDetectorConfig,
)
from skellytracker.core.detectors.keypoint_detectors.mediapipe import (
    MediapipeHandDetectorConfig,
    MediapipePoseDetectorConfig,
    MediapipePoseModelComplexity,
)
from skellytracker.core.detectors.object_detectors.keypoint_bbox import (
    KeypointBoundingBoxDetectorConfig,
)
from skellytracker.core.detectors.object_detectors.yolox.yolox_person_detector import (
    YoloxPersonDetectorConfig,
)
from skellytracker.core.tracker.tracker_factory import (
    _collect_required_sessions,
    build_multi_person_tracker,
    build_sessions,
    build_tracker,
)
from skellytracker.core.tracker.tracker_state import TrackerState


@pytest.fixture
def image() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestCollectRequiredSessions:
    def test_collects_mixed_backend_requirements(self):
        root = DetectionStageConfig(
            name="body",
            object_detector=YoloxPersonDetectorConfig(),
            keypoint_detectors=[ArucoDetectorConfig()],
            children=[
                DetectionStageConfig(
                    name="pose",
                    keypoint_detectors=[MediapipePoseDetectorConfig()],
                )
            ],
        )
        required = _collect_required_sessions([root])

        assert required.needs_mediapipe is True
        assert required.needs_cpu is True
        assert list(required.onnx_model_specs) == ["yolox-m"]

    def test_duplicate_onnx_model_merged_once(self):
        root = DetectionStageConfig(
            name="body",
            object_detector=YoloxPersonDetectorConfig(model_name="yolox-m"),
            children=[
                DetectionStageConfig(
                    name="body2",
                    object_detector=YoloxPersonDetectorConfig(model_name="yolox-m"),
                )
            ],
        )
        required = _collect_required_sessions([root])

        assert list(required.onnx_model_specs) == ["yolox-m"]

    def test_missing_model_spec_raises_clear_error(self):
        class _NoModelSpecConfig(ObjectDetectorConfig):
            detector_type: str = "fake_onnx_no_model_spec"
            session_backend: str = "onnx"
            model_name: str = "whatever"

        class _NoModelSpecDetector:
            pass

        OBJECT_DETECTOR_REGISTRY["fake_onnx_no_model_spec"] = _NoModelSpecDetector
        try:
            root = DetectionStageConfig(name="body", object_detector=_NoModelSpecConfig())
            with pytest.raises(TypeError, match="model_spec"):
                _collect_required_sessions([root])
        finally:
            del OBJECT_DETECTOR_REGISTRY["fake_onnx_no_model_spec"]


class TestBuildSessionsOnnxOverrides:
    def test_batch_size_override_does_not_raise_duplicate_kwarg(self, monkeypatch):
        from skellytracker.core.sessions.cpu_session import CpuSession
        from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig

        captured: dict[str, OnnxSessionConfig] = {}

        def _fake_create(cls, config):
            captured["config"] = config
            return CpuSession()

        monkeypatch.setattr(OnnxSession, "create", classmethod(_fake_create))

        config = TrackerConfig(
            stages=[DetectionStageConfig(name="body", object_detector=YoloxPersonDetectorConfig())]
        )
        sessions = build_sessions(config, onnx_overrides={"batch_size": 4})

        assert captured["config"].batch_size == 4
        assert sessions["onnx"] is not None


class TestBuildTracker:
    def test_build_tracker_end_to_end_mixed_mediapipe_and_cpu(self, image):
        hand_stage = DetectionStageConfig(
            name="right_hand",
            object_detector=KeypointBoundingBoxDetectorConfig(
                center_keypoint_names=("right_wrist",),
            ),
            keypoint_detectors=[MediapipeHandDetectorConfig(num_hands=1)],
        )
        body_stage = DetectionStageConfig(
            name="body",
            keypoint_detectors=[
                MediapipePoseDetectorConfig(model_complexity=MediapipePoseModelComplexity.LITE)
            ],
            children=[hand_stage],
        )
        config = TrackerConfig(stages=[body_stage])

        tracker = build_tracker(config)
        try:
            assert set(tracker.sessions) == {"mediapipe", "cpu"}
            observation, _ = tracker.process_image(image, frame_number=0, state=TrackerState.empty())
            assert "body" in observation.stages
        finally:
            tracker.close()

    def test_build_multi_person_tracker_rejects_multiple_roots(self):
        config = TrackerConfig(
            stages=[
                DetectionStageConfig(
                    name="a",
                    object_detector=KeypointBoundingBoxDetectorConfig(
                        center_keypoint_names=("right_wrist",)
                    ),
                ),
                DetectionStageConfig(
                    name="b",
                    object_detector=KeypointBoundingBoxDetectorConfig(
                        center_keypoint_names=("right_wrist",)
                    ),
                ),
            ]
        )

        with pytest.raises(ValueError, match="exactly one top-level stage"):
            build_multi_person_tracker(config)
