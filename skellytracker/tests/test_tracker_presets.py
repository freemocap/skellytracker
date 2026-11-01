"""Standalone composition contracts: no application or reconstruction imports."""
import numpy as np
import pytest

from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.data_primitives.observation import Observation, StageObservation
from skellytracker.core.io.mapping_paths import RTMPOSE_BODY_MAPPING, MEDIAPIPE_HAND_MAPPING
from skellytracker.core.io.tracker_profiles import tracker_profile
from skellytracker.core.tracker.keypoint_assembly import PreparedKeypointAssembly
from skellytracker.core.tracker.tracker_presets import combined_tracker_config
from skellytracker.core.tracker.tracker_factory import tracker_session_requests


def observation_for(config, *, missing=None):
    sources = config.keypoint_assembly.sources
    stages = []
    for index, source in enumerate(sources):
        count = len(source.source_names)
        points = Keypoints(source.source_names, np.full((count, 3), float(index + 1)), np.ones(count))
        stages.append(StageObservation(source.stage_path[-1], keypoints=points))
    body = stages[0]
    body.children = {stage.name: stage for stage in stages[1:] if stage.name != missing}
    return Observation(0, (480, 640), {"body": body, "charuco": StageObservation("charuco")})


def test_combined_schema_mapping_and_resources():
    config = combined_tracker_config()
    profile = tracker_profile("rtmw_mediapipe_hands")
    assert profile.mapping_paths == (RTMPOSE_BODY_MAPPING, MEDIAPIPE_HAND_MAPPING)
    assert len(profile.load_mapping().mappings) == len(profile.mapping_paths)
    assert "right_hand_index_finger_mcp" in profile.keypoint_names
    assert "right_hand_forefinger1" not in profile.keypoint_names
    assert "right_wrist" in profile.keypoint_names
    assert "right_hand_wrist" in profile.keypoint_names
    assert set(sum(profile.connections, ())).issubset(profile.keypoint_names)
    for child, side in zip(config.stages[0].children, ("right", "left")):
        assert child.object_detector.center_keypoint_names == (f"{side}_wrist",)
        assert child.object_detector.scale_keypoint_pairs == ((f"{side}_elbow", f"{side}_wrist"),)
        assert child.keypoint_detectors[0].assumed_handedness == side
    requests = tracker_session_requests(config, batch_size=3, onnx_overrides={"execution_provider": "cpu"})
    assert {r.config.backend for r in requests} == {"onnx", "mediapipe", "cpu"}
    onnx = next(r.config for r in requests if r.config.backend == "onnx")
    assert {m.name for m in onnx.models} == {"yolox-m", "rtmw-x-l_256x192"}


@pytest.mark.parametrize("missing", [None, "right_hand", "left_hand"])
def test_dense_assembly_missing_hands_and_storage(missing):
    config = combined_tracker_config()
    assembly = PreparedKeypointAssembly.prepare(config.keypoint_assembly)
    observation = observation_for(config, missing=missing)
    board = observation.stages["charuco"]
    original = observation.stages["body"].keypoints.xyz.copy()
    assembly.apply(observation)
    output = observation.stages["body"].keypoints
    assert output.names == tracker_profile("rtmw_mediapipe_hands").keypoint_names
    assert not observation.stages["body"].children
    assert observation.stages["charuco"] is board
    for source, (indices, destination) in zip(config.keypoint_assembly.sources, assembly.selections):
        if source.stage_path[-1] == missing:
            assert np.isnan(output.xyz[destination]).all()
            assert not output.visibility[destination].any()
        elif len(source.stage_path) == 1:
            np.testing.assert_array_equal(output.xyz[destination], original[indices])
        else:
            assert np.isfinite(output.xyz[destination]).all()
    retained = output.xyz.copy()
    next_observation = observation_for(config)
    assembly.apply(next_observation)
    next_observation.stages["body"].keypoints.xyz[:] = 100
    np.testing.assert_array_equal(output.xyz, retained)
    assert not np.shares_memory(output.xyz, next_observation.stages["body"].keypoints.xyz)


def test_schema_mismatch_fails_before_mutating_observation():
    config = combined_tracker_config()
    observation = observation_for(config)
    body = observation.stages["body"]
    original = body.keypoints
    body.children["right_hand"].keypoints = Keypoints.empty(("undeclared",))
    with pytest.raises(ValueError, match="schema mismatch"):
        PreparedKeypointAssembly.prepare(config.keypoint_assembly).apply(observation)
    assert body.keypoints is original
    assert "right_hand" in body.children


def test_prepared_keypoints_validate_values_without_rebuilding_metadata():
    template = Keypoints.empty(("example_keypoint",))
    values = template.with_values(xyz=np.ones_like(template.xyz), visibility=np.ones_like(template.visibility))
    assert values.index_of("example_keypoint") == 0
    assert values._name_to_idx is template._name_to_idx
    with pytest.raises(ValueError, match="shape"):
        template.with_values(xyz=np.ones((2, 3)), visibility=np.ones(2))


def test_tracker_single_and_batch_apply_identical_assembly():
    from unittest.mock import MagicMock, patch
    from skellytracker.core.tracker.tracker import Tracker
    from skellytracker.core.tracker.tracker_state import StageState, TrackerState
    from skellytracker.core.tracker.detection_stage import DetectionStage

    config = combined_tracker_config()
    stage = MagicMock(spec=DetectionStage, name="body_stage")
    stage.name = "body"
    stage.run.side_effect = lambda *args, **kwargs: (observation_for(config).stages["body"], StageState())
    stage.run_batch.side_effect = lambda images, *args: (
        {camera: observation_for(config).stages["body"] for camera in images},
        {camera: StageState() for camera in images})
    with patch.object(DetectionStage, "create", return_value=stage):
        tracker = Tracker.create_with_shared_sessions(config=config, sessions={})
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    single, _ = tracker.process_image(frame, 0, TrackerState())
    batch, _ = tracker.process_batch({"camera_a": frame, "camera_b": frame}, 0, {})
    for observation in batch.values():
        assert observation.stages["body"].keypoints.names == single.stages["body"].keypoints.names
        np.testing.assert_array_equal(observation.stages["body"].keypoints.xyz, single.stages["body"].keypoints.xyz)
    tracker.close()
    stage.close.assert_called_once()


def test_mediapipe_recipe_assembles_its_declared_schema_standalone():
    from skellytracker.core.tracker.tracker_presets import build_mediapipe_tracker_config
    config = build_mediapipe_tracker_config()
    assembly = PreparedKeypointAssembly.prepare(config.keypoint_assembly)
    observation = observation_for(config, missing="right_hand")
    assembly.apply(observation)
    assert observation.stages["body"].keypoints.names == tracker_profile("mediapipe").keypoint_names
    assert not observation.stages["body"].children


def test_assembly_requires_declared_source_stages():
    from skellytracker.core.config.tracker_config import TrackerConfig
    config = combined_tracker_config()
    config.stages[0].children.clear()
    with pytest.raises(ValueError, match="stage path"):
        TrackerConfig(stages=config.stages, keypoint_assembly=config.keypoint_assembly)


def test_existing_rtmw_recipe_preserves_public_schema():
    from skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.rtmw_wholebody_detector import RTMWWholebodyDetector
    assert tracker_profile("rtmpose").keypoint_names == RTMWWholebodyDetector.keypoint_names()
    assert tracker_profile("rtmpose").connections == RTMWWholebodyDetector.connections()
