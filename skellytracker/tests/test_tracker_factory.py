from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.config.detector_configs import KeypointDetectorConfig
from skellytracker.core.detectors.detector_base_classes import (
    KEYPOINT_DETECTOR_REGISTRY,
)
from skellytracker.core.detectors.keypoint_detectors.charuco import (
    CharucoDetectorConfig,
)
from skellytracker.core.detectors.keypoint_detectors.rtmpose import (
    RTMPoseDetectorConfig,
)
from skellytracker.core.detectors.keypoint_detectors.mediapipe.hands.mediapipe_hand_detector import (
    MediapipeHandDetectorConfig,
)
from skellytracker.core.detectors.object_detectors.yolox import (
    YoloxPersonDetectorConfig,
)
from skellytracker.core.sessions.cpu_session import CpuSession
from skellytracker.core.sessions.mediapipe_session import MediaPipeSession
from skellytracker.core.sessions.onnx_model_spec import OnnxModelSpec
from skellytracker.core.sessions.onnx_session import OnnxSession
from skellytracker.core.sessions.model_registry import ModelSource
from skellytracker.core.sessions.shared_sessions import SharedSessionPool
from skellytracker.core.tracker.tracker import Tracker
from skellytracker.core.tracker.detection_stage import DetectionStage
from skellytracker.core.tracker.multi_person_tracker import MultiPersonTracker
from skellytracker.core.tracker.tracker_factory import (
    build_sessions,
    build_tracker,
    collect_required_sessions,
    tracker_session_requests,
)


def mixed_config():
    return TrackerConfig(
        stages=[
            DetectionStageConfig(
                name="body",
                object_detector=YoloxPersonDetectorConfig(),
                keypoint_detectors=[RTMPoseDetectorConfig()],
                children=[
                    DetectionStageConfig(
                        name="hands", keypoint_detectors=[MediapipeHandDetectorConfig()]
                    ),
                    DetectionStageConfig(
                        name="board", keypoint_detectors=[CharucoDetectorConfig()]
                    ),
                ],
            )
        ]
    )


def test_mixed_configuration_discovers_every_nested_backend_without_allocating():
    with (
        patch.object(
            OnnxSession, "create", side_effect=AssertionError("no allocation")
        ),
        patch.object(
            MediaPipeSession, "create", side_effect=AssertionError("no allocation")
        ),
    ):
        requests = tracker_session_requests(
            mixed_config(), batch_size=3, onnx_overrides={"execution_provider": "cuda"}
        )
    assert [request.config.backend for request in requests] == [
        "cpu",
        "mediapipe",
        "onnx",
    ]
    assert requests[-1].config.batch_size == 3
    assert requests[-1].config.execution_provider == "cuda"
    assert [model.name for model in requests[-1].config.models] == [
        "rtmw-x-l_256x192",
        "yolox-m",
    ]


def test_new_registered_model_needs_no_factory_branch():
    class CustomConfig(KeypointDetectorConfig):
        detector_type: str = "custom_pose"
        session_backend: str = "onnx"
        model_name: str = "custom_model"

    descriptor = OnnxModelSpec(
        name="custom_model",
        source=ModelSource(local_path="model.onnx"),
        input_size=(32, 32),
    )
    with patch.dict(
        KEYPOINT_DETECTOR_REGISTRY,
        {"custom_pose": SimpleNamespace(model_spec=lambda name: descriptor)},
    ):
        config = TrackerConfig(
            stages=[
                DetectionStageConfig(
                    name="custom_stage", keypoint_detectors=[CustomConfig()]
                )
            ]
        )
        assert collect_required_sessions(config).onnx_model_specs == {
            "custom_model": descriptor
        }


def test_conflicting_model_names_are_rejected():
    class CustomConfig(KeypointDetectorConfig):
        detector_type: str = "custom_pose"
        session_backend: str = "onnx"
        model_name: str

    def model_spec(name):
        return OnnxModelSpec(
            name="same_name", source=ModelSource(local_path=name), input_size=(32, 32)
        )

    with patch.dict(
        KEYPOINT_DETECTOR_REGISTRY,
        {"custom_pose": SimpleNamespace(model_spec=model_spec)},
    ):
        config = TrackerConfig(
            stages=[
                DetectionStageConfig(
                    name="custom_stage",
                    keypoint_detectors=[
                        CustomConfig(model_name="first.onnx"),
                        CustomConfig(model_name="second.onnx"),
                    ],
                )
            ]
        )
        with pytest.raises(ValueError, match="Conflicting"):
            collect_required_sessions(config)


def test_partial_session_creation_failure_closes_created_sessions():
    cpu = CpuSession()
    with (
        patch.object(CpuSession, "create", return_value=cpu),
        patch.object(CpuSession, "close") as close,
        patch.object(
            MediaPipeSession, "create", side_effect=RuntimeError("failed model")
        ),
    ):
        with pytest.raises(RuntimeError, match="failed model"):
            build_sessions(mixed_config())
    close.assert_called_once()


def test_tracker_construction_failure_closes_all_allocated_sessions():
    config = TrackerConfig(
        stages=[
            DetectionStageConfig(
                name="board", keypoint_detectors=[CharucoDetectorConfig()]
            )
        ]
    )
    with (
        patch.object(Tracker, "create", side_effect=RuntimeError("failed detector")),
        patch.object(CpuSession, "close") as close,
    ):
        with pytest.raises(RuntimeError, match="failed detector"):
            build_tracker(config)
    close.assert_called_once()


def test_factory_requests_share_sessions_but_keep_tracker_state_separate():
    config = TrackerConfig(
        stages=[
            DetectionStageConfig(
                name="board", keypoint_detectors=[CharucoDetectorConfig()]
            )
        ]
    )
    requests = tracker_session_requests(config)
    pool = SharedSessionPool()
    try:
        first = pool.create_tracker(config=config, requests=requests)
        second = pool.create_tracker(config=config, requests=requests)
        assert first.tracker is not second.tracker
        assert first._entries[0].session is second._entries[0].session
        first.close()
        assert second._entries[0].references == 1
        second.close()
        assert not pool._entries
    finally:
        pool.close()


def test_undeclared_backend_is_rejected_before_allocation():
    config = TrackerConfig(
        stages=[
            DetectionStageConfig(
                name="unknown_stage",
                keypoint_detectors=[
                    KeypointDetectorConfig(
                        detector_type="unknown", session_backend="gpu_unknown"
                    )
                ],
            )
        ]
    )
    with patch.dict(KEYPOINT_DETECTOR_REGISTRY, {"unknown": object}):
        with pytest.raises(ValueError, match="Unsupported session backend"):
            tracker_session_requests(config)


def test_partial_detector_construction_closes_earlier_detectors():
    first = MagicMock()
    config = DetectionStageConfig(
        name="board",
        keypoint_detectors=[CharucoDetectorConfig(), CharucoDetectorConfig()],
    )
    with patch(
        "skellytracker.core.tracker.detection_stage.build_keypoint_detector",
        side_effect=[first, RuntimeError("failed detector")],
    ):
        with pytest.raises(RuntimeError, match="failed detector"):
            DetectionStage.create(config, {})
    first.close.assert_called_once()


@pytest.mark.parametrize("tracker_class", [Tracker, MultiPersonTracker])
def test_partial_stage_construction_closes_earlier_stages(tracker_class):
    first = MagicMock()
    config = TrackerConfig(
        stages=[DetectionStageConfig(name="first"), DetectionStageConfig(name="second")]
    )
    with patch.object(
        DetectionStage, "create", side_effect=[first, RuntimeError("failed stage")]
    ):
        with pytest.raises(RuntimeError, match="failed stage"):
            tracker_class.create(config, {})
    first.close.assert_called_once()


def test_actual_mixed_sessions_construct_and_release():
    sessions = build_sessions(
        mixed_config(),
        onnx_overrides={"execution_provider": "cpu"},
        mediapipe_overrides={"disable_gpu": True},
    )
    try:
        assert set(sessions) == {"onnx", "mediapipe", "cpu"}
        assert len(sessions["onnx"]._sessions) == 2
    finally:
        for session in sessions.values():
            session.close()
    assert not sessions["onnx"]._sessions
