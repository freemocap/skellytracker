"""Factory/session ownership and stage parity for dependent model detectors."""

from dataclasses import dataclass, field
from unittest.mock import Mock

import numpy as np
import pytest

from skellytracker.core.config.detector_configs import KeypointDetectorConfig
from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.data_primitives import Keypoints
from skellytracker.core.detectors.detector_base_classes import (
    KEYPOINT_DETECTOR_REGISTRY,
    KeypointDetector,
)
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.object_detectors.keypoint_bbox import (
    KeypointBoundingBoxDetectorConfig,
)
from skellytracker.core.sessions.cpu_session import CpuSession
from skellytracker.core.sessions.onnx_model_spec import OnnxModelSpec
from skellytracker.core.sessions.onnx_session import OnnxSession
from skellytracker.core.sessions.model_registry import ModelSource
from skellytracker.core.tracker.tracker_factory import (
    collect_required_sessions,
    tracker_session_requests,
)
from skellytracker.core.tracker.detection_stage import DetectionStage
from skellytracker.core.tracker.tracker_state import StageState


class DependentConfig(KeypointDetectorConfig):
    detector_type: str = "test_dependent_models"
    session_backend: str = "onnx"
    first_source: str = "first.onnx"


@dataclass
class DependentDetector(KeypointDetector):
    config: DependentConfig
    _point_names: tuple[str, ...] = ("test_point",)
    calls: list = field(default_factory=list)
    resets: int = 0
    closed: bool = False

    @classmethod
    def model_specs(cls, config):
        return (
            OnnxModelSpec(
                "first_model", ModelSource(local_path=config.first_source), (8, 8)
            ),
            OnnxModelSpec(
                "second_model", ModelSource(local_path="second.onnx"), (8, 8)
            ),
        )

    @classmethod
    def create(cls, config, session):
        return cls(config=config, session=session)

    def preprocess(self, image):
        raise AssertionError("Dependent inference must bypass the one-model path")

    def postprocess(self, raw, metadata):
        raise AssertionError("Dependent inference decodes both models itself")

    def detect(self, image, context=None):
        return self.detect_batch({"single": image}, context)["single"]

    def detect_batch(self, images, context=None, *, executor=None):
        self.calls.append((tuple(images), context.timestamp_ms, executor))
        return {
            camera: Keypoints(
                names=self._point_names,
                xyz=np.full((1, 3), float(image[0, 0, 0])),
                visibility=np.ones(1),
            )
            for camera, image in images.items()
        }

    def reset_temporal_state(self):
        self.resets += 1

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def register_detector(monkeypatch):
    monkeypatch.setitem(
        KEYPOINT_DETECTOR_REGISTRY, "test_dependent_models", DependentDetector
    )


def test_plural_requirements_deduplicate_and_keep_provider_request():
    config = TrackerConfig(
        stages=[
            DetectionStageConfig(name="first", keypoint_detectors=[DependentConfig()]),
            DetectionStageConfig(name="second", keypoint_detectors=[DependentConfig()]),
        ]
    )
    requirements = collect_required_sessions(config)
    assert set(requirements.onnx_model_specs) == {"first_model", "second_model"}
    (request,) = tracker_session_requests(
        config, batch_size=3, onnx_overrides={"execution_provider": "cpu"}
    )
    assert request.config.batch_size == 3 and request.config.execution_provider == "cpu"
    assert len(request.config.models) == 2
    config.stages[1].keypoint_detectors[0].first_source = "conflicting.onnx"
    with pytest.raises(ValueError, match="Conflicting"):
        collect_required_sessions(config)


@pytest.mark.parametrize("descriptors", [(), [], (object(),)])
def test_plural_requirements_reject_invalid_declarations(monkeypatch, descriptors):
    monkeypatch.setattr(DependentDetector, "model_specs", lambda config: descriptors)
    config = TrackerConfig(
        stages=[
            DetectionStageConfig(name="test", keypoint_detectors=[DependentConfig()])
        ]
    )
    with pytest.raises(TypeError):
        collect_required_sessions(config)


def test_stage_preserves_camera_identity_translation_timestamps_and_output_ownership():
    session = OnnxSession()
    session.close = Mock()
    config = DetectionStageConfig(
        name="hand",
        keypoint_detectors=[DependentConfig()],
        object_detector=KeypointBoundingBoxDetectorConfig(
            center_keypoint_names=("center",),
            scale_keypoint_pairs=(("center", "tip"),),
            scale_factor=2.0,
            min_box_size_px=8.0,
        ),
    )
    stage = DetectionStage.create(config, {"onnx": session, "cpu": CpuSession()})
    parent = Keypoints(
        names=("center", "tip"),
        xyz=np.array([[20.0, 20.0, 0.0], [25.0, 20.0, 0.0]]),
        visibility=np.ones(2),
    )
    retained = []
    for count in (1, 3, 2, 1):
        images = {
            f"camera_{index}": np.full((50, 50, 3), index + 1, dtype=np.uint8)
            for index in range(count)
        }
        context = DetectionContext(timestamp_ms=1000 + count)
        observations, states = stage.run_batch(
            images,
            {},
            context,
            parent_keypoints_per_cam={key: parent for key in images},
        )
        for key, image in images.items():
            single, _ = stage.run(
                image, StageState(), context=context, parent_keypoints=parent
            )
            np.testing.assert_array_equal(
                single.keypoints.xyz, observations[key].keypoints.xyz
            )
            (box,) = observations[key].bounding_boxes
            np.testing.assert_allclose(
                observations[key].keypoints.xyz[0],
                (image[0, 0, 0] + box.x1, image[0, 0, 0] + box.y1, image[0, 0, 0]),
            )
            assert states[key].last_timestamp_ms == context.timestamp_ms
            retained.append(
                (observations[key].keypoints, observations[key].keypoints.xyz.copy())
            )
    missing = Keypoints.empty(parent.names)
    calls = len(stage.keypoint_detectors[0].calls)
    observations, states = stage.run_batch(
        images, {}, context, parent_keypoints_per_cam={key: missing for key in images}
    )
    assert len(stage.keypoint_detectors[0].calls) == calls
    assert all(obs.keypoints.n_valid == 0 for obs in observations.values())
    assert all(state.consecutive_misses == [1] for state in states.values())
    stage.reset_temporal_state()
    assert stage.keypoint_detectors[0].resets == 1
    stage.close()
    assert stage.keypoint_detectors[0].closed
    session.close.assert_not_called()  # Borrowed sessions belong to Tracker/the pool.
    for result, original in retained:
        np.testing.assert_array_equal(result.xyz, original)


def test_stage_rejects_camera_identity_loss(monkeypatch):
    stage = DetectionStage.create(
        DetectionStageConfig(name="test", keypoint_detectors=[DependentConfig()]),
        {"onnx": OnnxSession()},
    )
    monkeypatch.setattr(
        stage.keypoint_detectors[0],
        "detect_batch",
        lambda *args, **kwargs: {"wrong_camera": Keypoints.empty(("test_point",))},
    )
    try:
        with pytest.raises(ValueError, match="camera identities"):
            stage.run_batch(
                {"camera": np.zeros((20, 20, 3), dtype=np.uint8)},
                {},
                DetectionContext(timestamp_ms=1000),
            )
    finally:
        stage.close()
