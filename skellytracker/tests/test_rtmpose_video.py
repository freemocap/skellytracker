"""Multiframe integration tests for the RTMPose tracker using real video.

Validates that the full YOLOX → RTMPose pipeline threads TrackerState correctly
across frames and produces consistent 133-keypoint output on real sequential images.

Skips automatically when onnxruntime is not installed or the test recording is
unavailable.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

import skellytracker.core.detectors.keypoint_detectors.rtmpose  # noqa: F401, E402
import skellytracker.core.detectors.object_detectors.yolox  # noqa: F401, E402

from skellytracker.core import (  # noqa: E402
    DetectionStageConfig,
    Tracker,
    TrackerConfig,
    TrackerState,
)
from skellytracker.core.detectors.keypoint_detectors.rtmpose import (  # noqa: E402
    RTMPoseDetectorConfig,
    RTMPoseKeypointDetector,
)
from skellytracker.core.detectors.object_detectors.yolox import (  # noqa: E402
    YoloxPersonDetector,
    YoloxPersonDetectorConfig,
)
from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig  # noqa: E402

pytestmark = pytest.mark.video

_N_FRAMES = 15


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
def onnx_session() -> OnnxSession:
    config = OnnxSessionConfig(
        batch_size=1,
        models=[
            YoloxPersonDetector.model_spec("yolox-m"),
            RTMPoseKeypointDetector.model_spec("rtmw-x-l_256x192"),
        ],
    )
    session = OnnxSession.create(config)
    yield session
    session.close()


class TestMultiFrameRTMPoseTracker:
    @pytest.fixture(scope="class")
    @classmethod
    def rtmpose_results(cls, test_video_path, onnx_session):
        frames = _load_video_frames(test_video_path, _N_FRAMES)
        if not frames:
            pytest.skip("No frames read from test video")
        config = TrackerConfig(
            stages=[
                DetectionStageConfig(
                    name="body",
                    object_detector=YoloxPersonDetectorConfig(),
                    keypoint_detectors=[RTMPoseDetectorConfig()],
                )
            ]
        )
        tracker = Tracker.create(config, {"onnx": onnx_session})
        observations = []
        states = [TrackerState()]
        for i, frame in enumerate(frames):
            obs, state = tracker.process_image(frame, frame_number=i, state=states[-1])
            observations.append(obs)
            states.append(state)
        return observations, states

    def test_observation_count(self, rtmpose_results):
        observations, _ = rtmpose_results
        assert len(observations) == _N_FRAMES

    def test_keypoint_shape_133(self, rtmpose_results):
        observations, _ = rtmpose_results
        for obs in observations:
            assert obs.stages["body"].keypoints.xyz.shape == (133, 3)

    def test_keypoint_shapes_consistent(self, rtmpose_results):
        observations, _ = rtmpose_results
        shapes = {obs.to_keypoints().xyz.shape for obs in observations}
        assert len(shapes) == 1, f"Inconsistent keypoint shapes across frames: {shapes}"

    def test_at_least_one_detection(self, rtmpose_results):
        observations, _ = rtmpose_results
        n_valid = sum(
            1 for obs in observations
            if obs.stages["body"].keypoints.n_valid > 0
        )
        assert n_valid > 0, "Expected at least one frame with a detected person"

    def test_state_populated_after_first_frame(self, rtmpose_results):
        _, states = rtmpose_results
        assert "body" in states[1].stage_states
