"""The opt-in experiment preserves existing crop/schema/assembly ownership."""

from skellytracker.core.detectors.keypoint_detectors.mediapipe_onnx.experimental_recipe import (
    experimental_tracker_config,
)
from skellytracker.core.tracker.tracker_presets import combined_tracker_config
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.io.tracker_profiles import tracker_profile


def test_experimental_app_preset_resolves_without_replacing_cpu_hybrid(monkeypatch, tmp_path):
    from skellytracker.core.tracker.tracker_presets import human_tracker_config
    monkeypatch.setenv("SKELLYTRACKER_HAND_CANDIDATE_DIR", str(tmp_path))
    candidate = human_tracker_config(preset="rtmw_onnx_hands", presence_confidence=0.7)
    baseline = human_tracker_config(preset="rtmw_mediapipe_hands")
    assert tracker_profile("rtmw_onnx_hands") is tracker_profile("rtmw_mediapipe_hands")
    assert candidate.keypoint_assembly == baseline.keypoint_assembly
    for child in candidate.stages[0].children:
        detector = child.keypoint_detectors[0]
        assert detector.session_backend == "onnx"
        assert detector.model_directory == tmp_path
        assert detector.min_hand_presence_confidence == 0.7
    assert all(child.keypoint_detectors[0].session_backend == "mediapipe"
               for child in baseline.stages[0].children)


def test_experimental_app_session_selection_requires_cpu_or_cuda(monkeypatch):
    import pytest
    from skellytracker.core.tracker.tracker_factory import tracker_session_requests
    candidate = experimental_tracker_config()
    for provider in (None, "trt", "coreml"):
        with pytest.raises(ValueError, match="explicit CPU or CUDA"):
            tracker_session_requests(candidate, batch_size=3, onnx_overrides={"execution_provider": provider})
    for provider in ("cpu", "cuda"):
        requests = tracker_session_requests(candidate, batch_size=3, onnx_overrides={"execution_provider": provider})
        assert len(next(request.config.models for request in requests if request.config.backend == "onnx")) == 4


def test_experiment_uses_shared_hybrid_schema_and_mapping_profile():
    production = combined_tracker_config()
    candidate = experimental_tracker_config()
    profile = tracker_profile(candidate.preset_name)
    assert profile is tracker_profile(production.preset_name)
    assert profile.load_mapping() is not None
    for child in candidate.stages[0].children:
        assert child.keypoint_detectors[0].detector_type == "experimental_mediapipe_onnx_hand"


def test_experiment_changes_hand_backend_without_replacing_crop_rules():
    settings = dict(detection_confidence=0.6, presence_confidence=0.7)
    production = combined_tracker_config(**settings)
    candidate = experimental_tracker_config(**settings)
    assert candidate.keypoint_assembly == production.keypoint_assembly
    assert candidate.stages[0].object_detector == production.stages[0].object_detector
    assert (
        candidate.stages[0].keypoint_detectors
        == production.stages[0].keypoint_detectors
    )
    for expected, actual in zip(
        production.stages[0].children, candidate.stages[0].children, strict=True
    ):
        assert expected.model_dump(exclude={"keypoint_detectors"}) == actual.model_dump(
            exclude={"keypoint_detectors"}
        )
        assert actual.object_detector.scale_factor == 3.0
        assert actual.keypoint_detectors[0].min_hand_detection_confidence == 0.6
        assert actual.keypoint_detectors[0].min_hand_presence_confidence == 0.7
        assert (
            actual.keypoint_detectors[0].assumed_handedness
            == expected.keypoint_detectors[0].assumed_handedness
        )
        assert expected.keypoint_detectors[0].session_backend == "mediapipe"
        assert actual.keypoint_detectors[0].session_backend == "onnx"
    restored = TrackerConfig.model_validate_json(candidate.model_dump_json())
    assert restored == candidate
