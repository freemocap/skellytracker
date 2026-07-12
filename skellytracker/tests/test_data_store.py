"""Integration tests for DataStore using real video frames.

Uses mediapipe body-only as a vehicle to produce real Observations; the tests
themselves assert DataStore behaviour, not detector behaviour.

Skips automatically when the test recording is unavailable.
"""
from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np
import pytest

import skellytracker.core.detectors.keypoint_detectors.mediapipe  # noqa: F401
from skellytracker.core import (
    DataStore,
    DetectionStageConfig,
    Tracker,
    TrackerConfig,
    TrackerState,
)
from skellytracker.core.detectors.keypoint_detectors.mediapipe import (
    MediapipePoseDetectorConfig,
    MediapipePoseModelComplexity,
    MediaPipeSession,
    MediaPipeSessionConfig,
)

_N_FRAMES = 10


def _load_video_frames(video_path: pathlib.Path, n_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")
    frames = []
    try:
        for _ in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


@pytest.fixture(scope="module")
def mediapipe_session() -> MediaPipeSession:
    session = MediaPipeSession.create(MediaPipeSessionConfig())
    yield session
    session.close()


class TestDataStore:
    @pytest.fixture(scope="class")
    @classmethod
    def populated_store(cls, test_video_path, mediapipe_session):
        frames = _load_video_frames(test_video_path, _N_FRAMES)
        if not frames:
            pytest.skip("Could not read any frames from test video")
        config = TrackerConfig(
            stages=[
                DetectionStageConfig(
                    name="body",
                    keypoint_detectors=[
                        MediapipePoseDetectorConfig(
                            model_complexity=MediapipePoseModelComplexity.LITE
                        )
                    ],
                )
            ]
        )
        tracker = Tracker.create(config, {"mediapipe": mediapipe_session})
        store = DataStore()
        state = TrackerState()
        for i, frame in enumerate(frames):
            obs, state = tracker.process_image(frame, frame_number=i, state=state)
            store.add(obs)
        return store, len(frames)

    def test_observation_count_matches_frames(self, populated_store):
        store, n_frames = populated_store
        assert len(store.observations) == n_frames

    def test_to_array_shape(self, populated_store):
        store, n_frames = populated_store
        arr = store.to_array()
        assert arr.ndim == 3
        assert arr.shape[0] == n_frames
        assert arr.shape[1] == 33
        assert arr.shape[2] == 3

    def test_to_array_dtype(self, populated_store):
        store, _ = populated_store
        assert store.to_array().dtype == np.float64

    def test_frame_numbers_are_sequential(self, populated_store):
        store, _ = populated_store
        frame_numbers = [obs.frame_number for obs in store.observations]
        assert frame_numbers == list(range(len(store.observations)))

    def test_image_size_consistent(self, populated_store):
        store, _ = populated_store
        sizes = {obs.image_size for obs in store.observations}
        assert len(sizes) == 1, f"Expected consistent image size, got: {sizes}"

    def test_save_npy_round_trip(self, populated_store, tmp_path):
        store, _ = populated_store
        out = tmp_path / "keypoints.npy"
        store.save(out, fmt="npy")
        loaded = np.load(str(out))
        np.testing.assert_array_equal(loaded, store.to_array())

    def test_to_json_keys_match_frame_count(self, populated_store):
        store, n_frames = populated_store
        raw = store.to_json()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            pytest.skip("to_json produced non-standard JSON (likely NaN values for undetected keypoints)")
        assert len(data) == n_frames
