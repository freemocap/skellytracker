"""Short standalone real-model checks on both prepared reference recordings.

Set SKELLYTRACKER_REFERENCE_ROOT to the prepared-recordings parent if needed.
These checks are not end-to-end Core acceptance or throughput benchmarks.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from skellytracker.core.io.tracker_profiles import tracker_profile
from skellytracker.core.tracker.tracker_factory import build_tracker
from skellytracker.core.tracker.tracker_presets import human_tracker_config
from skellytracker.core.tracker.tracker_state import TrackerState


@pytest.mark.video
@pytest.mark.parametrize("preset", ["rtmw_mediapipe_hands", "mediapipe"])
@pytest.mark.parametrize("dataset", ["freemocap_test_data", "freemocap_sample_data"])
@pytest.mark.parametrize("mode", ["single", "batch"])
def test_combined_reference_recording(dataset, mode, preset, record_property):
    root = Path(os.environ.get("SKELLYTRACKER_REFERENCE_ROOT",
                               str(Path.home() / "freemocap_data/testing/prepared")))
    paths = sorted((root / dataset / "current/recordings" / dataset / "synchronized_videos").glob("*.mp4"))
    if not paths:
        pytest.skip(f"Prepared reference recording unavailable: {root / dataset}")
    profile = tracker_profile(preset)
    hand_indices = np.asarray([i for i, name in enumerate(profile.keypoint_names)
                               if name.startswith(("left_hand_", "right_hand_"))])
    tracker = build_tracker(human_tracker_config(preset=preset), batch_size=len(paths) if mode == "batch" else 1,
                            onnx_overrides={"execution_provider": "cpu"},
                            mediapipe_overrides={"disable_gpu": True})
    retained_points = []
    total_observations = hand_points = visible_points = 0
    try:
        if mode == "single":
            for path in paths:
                tracker.reset_temporal_state()
                state = TrackerState()
                capture = cv2.VideoCapture(str(path))
                try:
                    assert capture.isOpened()
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    assert np.isfinite(fps) and fps > 0
                    for number in range(12):
                        ok, frame = capture.read()
                        assert ok
                        observation, state = tracker.process_image(frame, number, state, timestamp_ms=round(number * 1000 / fps))
                        points = observation.stages["body"].keypoints
                        assert points.names == profile.keypoint_names
                        if number == 0:
                            retained_points.append((points, points.xyz.copy(), points.visibility.copy()))
                        hand_points += int(np.isfinite(points.xy[hand_indices]).all(axis=1).sum())
                        visible_points += int(np.isfinite(points.xy).all(axis=1).sum())
                        total_observations += 1
                finally:
                    capture.release()
        else:
            captures = [cv2.VideoCapture(str(path)) for path in paths]
            rates = [capture.get(cv2.CAP_PROP_FPS) for capture in captures]
            assert all(np.isfinite(rate) and rate > 0 for rate in rates)
            assert np.allclose(rates, rates[0])
            fps = rates[0]
            states = {}
            try:
                for number in range(12):
                    images = {}
                    for path, capture in zip(paths, captures):
                        ok, frame = capture.read()
                        assert ok
                        images[path.stem] = frame
                    observations, states = tracker.process_batch(images, number, states, timestamp_ms=round(number * 1000 / fps))
                    for observation in observations.values():
                        points = observation.stages["body"].keypoints
                        assert points.names == profile.keypoint_names
                        if number == 0:
                            retained_points.append((points, points.xyz.copy(), points.visibility.copy()))
                        for child in observation.stages['body'].children.values():
                            assert child.keypoints is None  # Assembled exactly once.
                            for box in child.bounding_boxes:
                                height, width = observation.image_size
                                assert 0 <= box.x1 <= box.x2 <= width
                                assert 0 <= box.y1 <= box.y2 <= height
                                assert all(value == int(value) for value in (box.x1, box.y1, box.x2, box.y2))
                        hand_points += int(np.isfinite(points.xy[hand_indices]).all(axis=1).sum())
                        visible_points += int(np.isfinite(points.xy).all(axis=1).sum())
                        total_observations += 1
            finally:
                for capture in captures:
                    capture.release()
    finally:
        tracker.close()
    for points, original_xyz, original_visibility in retained_points:
        np.testing.assert_array_equal(points.xyz, original_xyz)
        np.testing.assert_array_equal(points.visibility, original_visibility)
    assert total_observations == 12 * len(paths)
    assert visible_points > 0
    record_property("observations", total_observations)
    record_property("finite_hand_keypoints", hand_points)
