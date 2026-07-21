"""Tests for the new mediapipe KeypointDetector implementations."""
from __future__ import annotations

import numpy as np
import pytest

import skellytracker.core.detectors.keypoint_detectors.mediapipe  # noqa: F401 — triggers registry side-effects
from skellytracker.core import (
    KEYPOINT_DETECTOR_REGISTRY,
    DetectionStageConfig,
    Tracker,
    TrackerConfig,
    TrackerState,
)
from skellytracker.core.detectors.keypoint_detectors.mediapipe import (
    MediapipeFaceDetectorConfig,
    MediapipeFaceKeypointDetector,
    MediapipeHandDetectorConfig,
    MediapipeHandKeypointDetector,
    MediapipePoseDetectorConfig,
    MediapipePoseKeypointDetector,
    MediapipePoseModelComplexity,
    MediaPipeSession,
    MediaPipeSessionConfig,
)


@pytest.fixture(scope="module")
def full_session() -> MediaPipeSession:
    session = MediaPipeSession.create(MediaPipeSessionConfig())
    yield session
    session.close()


class TestRegistry:
    def test_pose_registered(self):
        assert "mediapipe_pose" in KEYPOINT_DETECTOR_REGISTRY
        assert KEYPOINT_DETECTOR_REGISTRY["mediapipe_pose"] is MediapipePoseKeypointDetector

    def test_hand_registered(self):
        assert "mediapipe_hand" in KEYPOINT_DETECTOR_REGISTRY
        assert KEYPOINT_DETECTOR_REGISTRY["mediapipe_hand"] is MediapipeHandKeypointDetector

    def test_face_registered(self):
        assert "mediapipe_face" in KEYPOINT_DETECTOR_REGISTRY
        assert KEYPOINT_DETECTOR_REGISTRY["mediapipe_face"] is MediapipeFaceKeypointDetector


class TestPoseDetector:
    def test_detect_returns_correct_shape(self, test_image, full_session):
        detector = MediapipePoseKeypointDetector.create(
            MediapipePoseDetectorConfig(model_complexity=MediapipePoseModelComplexity.LITE), full_session
        )
        kpts = detector.detect(test_image)
        assert kpts.xyz.shape == (33, 3)
        assert kpts.visibility.shape == (33,)
        assert len(kpts.names) == 33

    def test_visibility_in_range(self, test_image, full_session):
        detector = MediapipePoseKeypointDetector.create(
            MediapipePoseDetectorConfig(model_complexity=MediapipePoseModelComplexity.LITE), full_session
        )
        kpts = detector.detect(test_image)
        assert np.all(kpts.visibility >= 0.0)
        assert np.all(kpts.visibility <= 1.0)

    def test_detection_on_real_image(self, test_image, full_session):
        detector = MediapipePoseKeypointDetector.create(
            MediapipePoseDetectorConfig(model_complexity=MediapipePoseModelComplexity.LITE), full_session
        )
        kpts = detector.detect(test_image)
        assert kpts.n_valid > 0, "Expected at least one detected landmark on test image"

    def test_empty_on_blank_image(self, full_session):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = MediapipePoseKeypointDetector.create(
            MediapipePoseDetectorConfig(model_complexity=MediapipePoseModelComplexity.LITE), full_session
        )
        kpts = detector.detect(blank)
        assert kpts.xyz.shape == (33, 3)
        assert np.all(np.isnan(kpts.xyz))
        assert np.all(kpts.visibility == 0.0)

    def test_point_names_include_key_landmarks(self, full_session):
        detector = MediapipePoseKeypointDetector.create(
            MediapipePoseDetectorConfig(model_complexity=MediapipePoseModelComplexity.LITE), full_session
        )
        assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)).has_name("nose")
        assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)).has_name("left_shoulder")


class TestHandDetector:
    def test_detect_returns_correct_shape(self, test_image, full_session):
        detector = MediapipeHandKeypointDetector.create(
            MediapipeHandDetectorConfig(), full_session
        )
        kpts = detector.detect(test_image)
        assert kpts.xyz.shape == (42, 3)
        assert kpts.visibility.shape == (42,)
        assert len(kpts.names) == 42

    def test_visibility_in_range(self, test_image, full_session):
        detector = MediapipeHandKeypointDetector.create(
            MediapipeHandDetectorConfig(), full_session
        )
        kpts = detector.detect(test_image)
        assert np.all(kpts.visibility >= 0.0)
        assert np.all(kpts.visibility <= 1.0)

    def test_point_names_have_prefixes(self, full_session):
        detector = MediapipeHandKeypointDetector.create(
            MediapipeHandDetectorConfig(), full_session
        )
        blank = np.zeros((10, 10, 3), dtype=np.uint8)
        kpts = detector.detect(blank)
        assert kpts.has_name("right_hand_wrist")
        assert kpts.has_name("left_hand_wrist")
        assert kpts.has_name("right_hand_thumb_tip")
        assert kpts.has_name("left_hand_pinky_tip")

    def test_undetected_hands_are_nan(self, full_session):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = MediapipeHandKeypointDetector.create(
            MediapipeHandDetectorConfig(), full_session
        )
        kpts = detector.detect(blank)
        assert np.all(np.isnan(kpts.xyz))
        assert np.all(kpts.visibility == 0.0)


class TestFaceDetector:
    def test_detect_returns_correct_shape(self, test_image, full_session):
        detector = MediapipeFaceKeypointDetector.create(
            MediapipeFaceDetectorConfig(), full_session
        )
        kpts = detector.detect(test_image)
        assert kpts.xyz.shape[1] == 3
        assert kpts.visibility.shape == (kpts.xyz.shape[0],)
        assert len(kpts.names) == kpts.xyz.shape[0]

    def test_visibility_in_range(self, test_image, full_session):
        detector = MediapipeFaceKeypointDetector.create(
            MediapipeFaceDetectorConfig(), full_session
        )
        kpts = detector.detect(test_image)
        assert np.all(kpts.visibility >= 0.0)
        assert np.all(kpts.visibility <= 1.0)

    def test_point_names_use_face_prefix(self, full_session):
        detector = MediapipeFaceKeypointDetector.create(
            MediapipeFaceDetectorConfig(), full_session
        )
        blank = np.zeros((10, 10, 3), dtype=np.uint8)
        kpts = detector.detect(blank)
        assert all(n.startswith("face_") for n in kpts.names)


class TestTrackerIntegration:
    def test_full_pipeline(self, test_image, full_session):
        config = TrackerConfig(
            stages=[
                DetectionStageConfig(
                    name="body",
                    keypoint_detectors=[MediapipePoseDetectorConfig(model_complexity=MediapipePoseModelComplexity.LITE)],
                ),
                DetectionStageConfig(
                    name="hands",
                    keypoint_detectors=[MediapipeHandDetectorConfig()],
                ),
                DetectionStageConfig(
                    name="face",
                    keypoint_detectors=[MediapipeFaceDetectorConfig()],
                ),
            ]
        )
        sessions = {"mediapipe": full_session}
        tracker = Tracker.create(config, sessions)
        state = TrackerState()

        observation, state = tracker.process_image(test_image, frame_number=0, state=state)

        assert "body" in observation.stages
        assert "hands" in observation.stages
        assert "face" in observation.stages
        assert observation.stages["body"].keypoints.has_name("nose")
        assert observation.stages["hands"].keypoints.has_name("right_hand_wrist")
        assert any(n.startswith("face_") for n in observation.stages["face"].keypoints.names)
