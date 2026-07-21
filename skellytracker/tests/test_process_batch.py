"""Tests for Tracker.process_batch — multi-camera batched inference.

Uses MediaPipe body (CPU, no GPU required) with 2 synthetic cameras
(same test image duplicated) to verify the batch API without needing
the test recording.
"""
from __future__ import annotations

import numpy as np
import pytest

import skellytracker.core.detectors.keypoint_detectors.mediapipe  # noqa: F401 — triggers registry side-effects

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


@pytest.fixture(scope="module")
def mediapipe_session() -> MediaPipeSession:
    # Use IMAGE mode so that the same landmarker can be called multiple times
    # with non-monotonic timestamps — required for multi-camera batch testing
    # where the same detector processes N cameras per frame.
    session = MediaPipeSession.create(MediaPipeSessionConfig(running_mode="image"))
    yield session
    session.close()


@pytest.fixture(scope="module")
def body_tracker(mediapipe_session: MediaPipeSession) -> Tracker:
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
    return Tracker.create(config, {"mediapipe": mediapipe_session})


def _two_cam_images(test_image: np.ndarray) -> dict[str, np.ndarray]:
    return {"cam0": test_image, "cam1": test_image}


class TestProcessBatch:
    def test_process_batch_returns_observation_per_camera(self, body_tracker, test_image):
        images = _two_cam_images(test_image)
        states: dict[str, TrackerState] = {}
        observations, updated_states = body_tracker.process_batch(
            images=images, frame_number=0, states=states
        )
        assert set(observations.keys()) == {"cam0", "cam1"}
        assert set(updated_states.keys()) == {"cam0", "cam1"}

    def test_process_batch_keypoint_shapes_consistent_across_cameras(self, body_tracker, test_image):
        images = _two_cam_images(test_image)
        states: dict[str, TrackerState] = {}
        observations, _ = body_tracker.process_batch(
            images=images, frame_number=0, states=states
        )
        kpts0 = observations["cam0"].stages["body"].keypoints
        kpts1 = observations["cam1"].stages["body"].keypoints
        assert kpts0 is not None and kpts1 is not None
        assert kpts0.xyz.shape == kpts1.xyz.shape
        assert kpts0.visibility.shape == kpts1.visibility.shape
        assert kpts0.names == kpts1.names

    def test_process_batch_states_updated_per_camera(self, body_tracker, test_image):
        images = _two_cam_images(test_image)
        states: dict[str, TrackerState] = {}
        _, updated_states = body_tracker.process_batch(
            images=images, frame_number=0, states=states
        )
        assert "body" in updated_states["cam0"].stage_states
        assert "body" in updated_states["cam1"].stage_states
        # States are independent objects per camera
        assert updated_states["cam0"] is not updated_states["cam1"]

    def test_process_batch_matches_sequential_process_image_shape(self, body_tracker, test_image):
        """Batch and sequential processing yield the same keypoint shape."""
        images = _two_cam_images(test_image)
        states_batch: dict[str, TrackerState] = {}
        batch_obs, _ = body_tracker.process_batch(
            images=images, frame_number=0, states=states_batch
        )

        # Process each camera independently
        seq_obs0, _ = body_tracker.process_image(
            image=test_image, frame_number=0, state=TrackerState()
        )
        seq_obs1, _ = body_tracker.process_image(
            image=test_image, frame_number=0, state=TrackerState()
        )

        batch_kpts_cam0 = batch_obs["cam0"].stages["body"].keypoints
        batch_kpts_cam1 = batch_obs["cam1"].stages["body"].keypoints
        seq_kpts0 = seq_obs0.stages["body"].keypoints
        seq_kpts1 = seq_obs1.stages["body"].keypoints

        assert batch_kpts_cam0 is not None
        assert batch_kpts_cam1 is not None
        assert seq_kpts0 is not None
        assert seq_kpts1 is not None

        assert batch_kpts_cam0.xyz.shape == seq_kpts0.xyz.shape
        assert batch_kpts_cam1.xyz.shape == seq_kpts1.xyz.shape

    def test_single_camera_process_batch_matches_process_image_shape(self, body_tracker, test_image):
        """1-element dict gives the same keypoint shape as process_image."""
        images = {"cam0": test_image}
        batch_obs, _ = body_tracker.process_batch(
            images=images, frame_number=0, states={}
        )
        seq_obs, _ = body_tracker.process_image(
            image=test_image, frame_number=0, state=TrackerState()
        )

        batch_kpts = batch_obs["cam0"].stages["body"].keypoints
        seq_kpts = seq_obs.stages["body"].keypoints

        assert batch_kpts is not None and seq_kpts is not None
        assert batch_kpts.xyz.shape == seq_kpts.xyz.shape
        assert batch_kpts.visibility.shape == seq_kpts.visibility.shape
        assert batch_kpts.names == seq_kpts.names

    def test_process_batch_empty_images_returns_empty(self, body_tracker):
        observations, updated_states = body_tracker.process_batch(
            images={}, frame_number=0, states={}
        )
        assert observations == {}
        assert updated_states == {}

    def test_process_batch_new_camera_gets_empty_initial_state(self, body_tracker, test_image):
        """Cameras not in states dict should start with empty TrackerState."""
        images = {"cam0": test_image}
        # Pass no initial states — camera should still work
        observations, updated_states = body_tracker.process_batch(
            images=images, frame_number=0, states={}
        )
        assert "cam0" in observations
        assert "cam0" in updated_states
        kpts = observations["cam0"].stages["body"].keypoints
        assert kpts is not None
        assert kpts.xyz.shape == (33, 3)
