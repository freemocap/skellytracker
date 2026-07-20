"""Multiframe integration tests for the CharucoDetector using real video.

The test recording begins with a charuco board in frame, giving real detections
across the first several frames.

Skips automatically when the test recording is unavailable.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pytest

from skellytracker.core.detectors.keypoint_detectors.charuco import (
    CharucoBoardDefinition,
    CharucoDetector,
    CharucoDetectorConfig,
)
from skellytracker.core.sessions.cpu_session import CpuSession, CpuSessionConfig

pytestmark = pytest.mark.video

_N_FRAMES = 30


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
def board_def() -> CharucoBoardDefinition:
    return CharucoBoardDefinition.create_test_data_7x5()


@pytest.fixture(scope="module")
def cpu_session() -> CpuSession:
    session = CpuSession.create(CpuSessionConfig())
    yield session
    session.close()


@pytest.fixture(scope="module")
def detector(board_def, cpu_session) -> CharucoDetector:
    return CharucoDetector.create(CharucoDetectorConfig(board=board_def), cpu_session)


class TestCharucoDetectorVideo:
    @pytest.fixture(scope="class")
    @classmethod
    def video_results(cls, test_video_path, detector, board_def):
        frames = _load_video_frames(test_video_path, _N_FRAMES)
        if not frames:
            pytest.skip("No frames read from test video")
        return [detector.detect(frame) for frame in frames], board_def

    def test_all_frames_produce_keypoints(self, video_results):
        results, board_def = video_results
        expected_n_aruco = len(results[0].names) - board_def.n_corners
        expected_total = board_def.n_corners + expected_n_aruco
        for kpts in results:
            assert kpts.xyz.shape == (expected_total, 3)

    def test_keypoint_shape_consistent_across_frames(self, video_results):
        results, _ = video_results
        shapes = {kpts.xyz.shape for kpts in results}
        assert len(shapes) == 1, f"Inconsistent shapes across frames: {shapes}"

    def test_at_least_one_frame_detects_corners(self, video_results):
        results, _ = video_results
        n_frames_with_detection = sum(1 for kpts in results if kpts.n_valid > 0)
        assert n_frames_with_detection > 0, "Expected charuco corners detected in at least one frame"

    def test_visibility_in_range(self, video_results):
        results, _ = video_results
        for kpts in results:
            assert np.all(kpts.visibility >= 0.0)
            assert np.all(kpts.visibility <= 1.0)

    def test_detected_points_have_finite_xy(self, video_results):
        results, _ = video_results
        for kpts in results:
            detected = kpts.visibility > 0.0
            assert np.all(np.isfinite(kpts.xyz[detected, :2]))

    def test_undetected_points_are_nan(self, video_results):
        results, _ = video_results
        for kpts in results:
            undetected = kpts.visibility == 0.0
            assert np.all(np.isnan(kpts.xyz[undetected, 0]))
