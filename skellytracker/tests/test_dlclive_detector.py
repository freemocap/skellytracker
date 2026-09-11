"""Tests for the DeepLabCut-Live (PyTorch) keypoint detector.

Skips automatically if dlclive is not installed. Full detect()/Tracker
integration requires a real exported PyTorch model, which isn't bundled or
downloadable the way YOLOX/RTMPose models are — those tests are gated behind
the SKELLYTRACKER_DLCLIVE_TEST_MODEL environment variable.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("dlclive", reason="dlclive not installed")

import skellytracker.core.detectors.keypoint_detectors.deeplabcut_live  # noqa: F401, E402
from skellytracker.core import (
    KEYPOINT_DETECTOR_REGISTRY,  # noqa: E402
    DetectionStageConfig,
    Tracker,
    TrackerConfig,
    TrackerState,
)
from skellytracker.core.detectors.keypoint_detectors.deeplabcut_live import (  # noqa: E402
    DLCLiveKeypointDetector,
    DLCLiveKeypointDetectorConfig,
)
from skellytracker.core.detectors.keypoint_detectors.deeplabcut_live.dlc_live_keypoint_detector import (
    _extract_bodypart_names,
)  # noqa: E402
from skellytracker.core.sessions.dlc_live_session import (  # noqa: E402
    DLCLiveSession,
    DLCLiveSessionConfig,
)

_MODEL_PATH = os.environ.get("SKELLYTRACKER_DLCLIVE_TEST_MODEL")

requires_real_model = pytest.mark.skipif(
    not _MODEL_PATH,
    reason="SKELLYTRACKER_DLCLIVE_TEST_MODEL not set; no real exported model available",
)


class TestRegistry:
    def test_dlclive_registered(self):
        assert "dlclive" in KEYPOINT_DETECTOR_REGISTRY
        assert KEYPOINT_DETECTOR_REGISTRY["dlclive"] is DLCLiveKeypointDetector


class TestConfig:
    def test_config_requires_model_path(self):
        with pytest.raises(ValidationError):
            DLCLiveKeypointDetectorConfig()

    def test_config_defaults(self):
        config = DLCLiveKeypointDetectorConfig(model_path="/fake/model.pt")
        assert config.detector_type == "dlclive"
        assert config.session_backend == "dlclive"
        assert config.confidence_threshold == 0.0
        assert config.dynamic == (False, 0.5, 10)
        assert config.resize is None

    def test_create_rejects_wrong_session_type(self):
        config = DLCLiveKeypointDetectorConfig(model_path="/fake/model.pt")

        class NotASession:
            pass

        with pytest.raises(TypeError):
            DLCLiveKeypointDetector.create(config, NotASession())


class TestExtractBodypartNames:
    def test_finds_names_at_metadata_bodyparts(self):
        cfg = {"metadata": {"bodyparts": ["nose", "left_ear", "right_ear"]}}
        assert _extract_bodypart_names(cfg) == ("nose", "left_ear", "right_ear")

    def test_finds_names_at_top_level_bodyparts(self):
        cfg = {"bodyparts": ["a", "b"]}
        assert _extract_bodypart_names(cfg) == ("a", "b")

    def test_finds_names_at_all_joints_names(self):
        cfg = {"all_joints_names": ["a", "b", "c"]}
        assert _extract_bodypart_names(cfg) == ("a", "b", "c")

    def test_raises_clear_error_when_missing(self):
        with pytest.raises(ValueError, match="Could not find bodypart"):
            _extract_bodypart_names({"unrelated": "value"})


@requires_real_model
class TestDLCLiveKeypointDetectorWithRealModel:
    @pytest.fixture(scope="class")
    def session(self) -> DLCLiveSession:
        session = DLCLiveSession.create(DLCLiveSessionConfig())
        yield session
        session.close()

    def test_detect_returns_keypoints(self, test_image, session):
        config = DLCLiveKeypointDetectorConfig(model_path=_MODEL_PATH)
        detector = DLCLiveKeypointDetector.create(config, session)
        kpts = detector.detect(test_image)
        assert kpts.xyz.shape[1] == 3
        assert kpts.visibility.shape[0] == kpts.xyz.shape[0]
        assert len(kpts.names) == kpts.xyz.shape[0]

    def test_visibility_in_range(self, test_image, session):
        config = DLCLiveKeypointDetectorConfig(model_path=_MODEL_PATH)
        detector = DLCLiveKeypointDetector.create(config, session)
        kpts = detector.detect(test_image)
        assert np.all(kpts.visibility >= 0.0)
        assert np.all(kpts.visibility <= 1.0)

    def test_full_pipeline(self, test_image, session):
        config = TrackerConfig(
            stages=[
                DetectionStageConfig(
                    name="dlclive",
                    keypoint_detectors=[
                        DLCLiveKeypointDetectorConfig(model_path=_MODEL_PATH)
                    ],
                )
            ]
        )
        tracker = Tracker.create(config, {"dlclive": session})
        state = TrackerState()

        observation, state = tracker.process_image(
            test_image, frame_number=0, state=state
        )

        assert "dlclive" in observation.stages
        assert observation.stages["dlclive"].keypoints is not None
