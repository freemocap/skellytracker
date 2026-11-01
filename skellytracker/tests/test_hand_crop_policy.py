"""Both presets crop the same hand geometry despite different keypoint names."""
import numpy as np
import pytest

from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.detectors.object_detectors.keypoint_bbox import KeypointBoundingBoxDetector
from skellytracker.core.io.tracker_profiles import tracker_profile
from skellytracker.core.tracker.tracker_presets import human_tracker_config


@pytest.mark.parametrize("preset", ["mediapipe", "rtmw_mediapipe_hands"])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("angle", [0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
def test_presets_center_and_cover_equivalent_hand_geometry(preset, side, angle):
    parent_preset = "rtmpose" if preset == "rtmw_mediapipe_hands" else "mediapipe"
    names = tracker_profile(parent_preset).keypoint_names
    parent = Keypoints.empty(names)
    finger_names = ((f"{side}_hand_forefinger4", f"{side}_hand_pinky_finger4")
                    if parent_preset == "rtmpose" else (f"{side}_index", f"{side}_pinky"))
    # Foreshortened forearm, fingers extending away from the wrist: the old
    # wrist/forearm crop excludes both fingertips in every orientation.
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    relative = np.array([[0, 0], [-10, 0], [45, -10], [35, 10]], dtype=float)
    coordinates = relative @ rotation.T + [240, 240]
    indices = [names.index(name) for name in (f"{side}_wrist", f"{side}_elbow", *finger_names)]
    parent.xyz[indices, :2] = coordinates
    parent.xyz[indices, 2] = 0
    parent.visibility[indices] = 1
    stage = next(child for child in human_tracker_config(preset=preset).stages[0].children
                 if child.name == f"{side}_hand")
    detector = KeypointBoundingBoxDetector(config=stage.object_detector)
    image = np.zeros((480, 480, 3), dtype=np.uint8)
    box, = detector.detect(image, parent_keypoints=parent)
    np.testing.assert_allclose(box.center, coordinates[2:].mean(axis=0))
    expected_size = 3 * np.linalg.norm(coordinates[2:] - coordinates[0], axis=1).max()
    assert box.width == pytest.approx(expected_size)
    assert box.height == pytest.approx(expected_size)
    assert np.all(coordinates[[0, 2, 3]] >= [box.x1, box.y1])
    assert np.all(coordinates[[0, 2, 3]] <= [box.x2, box.y2])
    # Missing one finger still yields a valid crop; no fingers means no invented crop.
    parent.xyz[indices[2]] = np.nan
    partial_box, = detector.detect(image, parent_keypoints=parent)
    np.testing.assert_allclose(partial_box.center, coordinates[3])
    parent.xyz[indices[3]] = np.nan
    assert detector.detect(image, parent_keypoints=parent) == []


def test_standalone_demo_uses_the_same_hand_stages(monkeypatch):
    from skellytracker.core.detectors.keypoint_detectors.mediapipe import run_demo
    from skellytracker.core.tracker.tracker_presets import mediapipe_hand_stage

    captured_configs = []
    monkeypatch.setattr(run_demo, "build_tracker", lambda config: captured_configs.append(config))
    manager_type = run_demo.DemoManager
    monkeypatch.setattr(run_demo, "DemoManager", lambda **kwargs: object.__new__(manager_type))
    run_demo.build_mediapipe_demo_with_cropped_hands()
    children = captured_configs[0].stages[0].children
    assert [child.model_dump() for child in children] == [
        mediapipe_hand_stage(side).model_dump() for side in ("left", "right")]


@pytest.mark.parametrize("kwargs", [{"side": "unknown"}, {"side": "left", "parent_preset": "unknown"}])
def test_unknown_hand_schema_is_rejected(kwargs):
    from skellytracker.core.tracker.tracker_presets import mediapipe_hand_stage
    with pytest.raises(ValueError):
        mediapipe_hand_stage(**kwargs)
