"""Multiframe integration tests for the mediapipe tracker using real video.

Validates that TrackerState threads correctly across frames and that the full
Tracker → DetectionStage → MediapipePoseKeypointDetector pipeline behaves
consistently on real sequential images.

Skips automatically when the test recording is unavailable.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pytest

import skellytracker.core.detectors.keypoint_detectors.mediapipe  # noqa: F401
from skellytracker.core import (
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

_N_FRAMES = 20


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


class TestMultiFrameMediapipeTracker:
    @pytest.fixture(scope="class")
    @classmethod
    def multiframe_results(cls, test_video_path, mediapipe_session):
        frames = _load_video_frames(test_video_path, _N_FRAMES)
        if not frames:
            pytest.skip("No frames read from test video")
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
        observations = []
        states = [TrackerState()]
        for i, frame in enumerate(frames):
            obs, state = tracker.process_image(frame, frame_number=i, state=states[-1])
            observations.append(obs)
            states.append(state)
        return observations, states, frames

    def test_all_frames_produce_observation(self, multiframe_results):
        observations, _, frames = multiframe_results
        assert len(observations) == len(frames)

    def test_state_is_populated_after_first_frame(self, multiframe_results):
        _, states, _ = multiframe_results
        assert "body" in states[1].stage_states

    def test_keypoint_shapes_consistent_across_frames(self, multiframe_results):
        observations, _, _ = multiframe_results
        shapes = {obs.to_keypoints().xyz.shape for obs in observations}
        assert len(shapes) == 1, f"Inconsistent keypoint shapes across frames: {shapes}"

    def test_keypoint_count_is_33(self, multiframe_results):
        observations, _, _ = multiframe_results
        for obs in observations:
            assert obs.stages["body"].keypoints.xyz.shape == (33, 3)

    def test_at_least_one_frame_has_valid_detection(self, multiframe_results):
        observations, _, _ = multiframe_results
        n_valid = sum(
            1 for obs in observations
            if obs.stages["body"].keypoints.n_valid > 0
        )
        assert n_valid > 0, "Expected at least one frame with a detected person"

    def test_state_accumulates_last_keypoints(self, multiframe_results):
        _, states, _ = multiframe_results
        body_state = states[-1].stage_states.get("body")
        assert body_state is not None
        assert body_state.last_keypoints is not None

    def test_frame_numbers_in_observations(self, multiframe_results):
        observations, _, _ = multiframe_results
        for i, obs in enumerate(observations):
            assert obs.frame_number == i
