"""Multiframe integration tests for the YOLOX person detector using real video.

Tests the object detector in isolation across multiple frames, verifying that
bounding boxes are valid and at least some frames detect the person present
in the test recording.

Skips automatically when onnxruntime is not installed or the test recording is
unavailable.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

import skellytracker.core.detectors.object_detectors.yolox  # noqa: F401, E402

from skellytracker.core.detectors.object_detectors.yolox import (  # noqa: E402
    YoloxPersonDetector,
    YoloxPersonDetectorConfig,
)
from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig  # noqa: E402

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
def onnx_session() -> OnnxSession:
    config = OnnxSessionConfig(
        batch_size=1,
        models=[YoloxPersonDetector.model_spec("yolox-m")],
    )
    session = OnnxSession.create(config)
    yield session
    session.close()


class TestYoloxPersonDetectorVideo:
    @pytest.fixture(scope="class")
    @classmethod
    def video_results(cls, test_video_path, onnx_session):
        frames = _load_video_frames(test_video_path, _N_FRAMES)
        if not frames:
            pytest.skip("No frames read from test video")
        detector = YoloxPersonDetector.create(YoloxPersonDetectorConfig(), onnx_session)
        return [detector.detect(frame) for frame in frames]

    def test_all_frames_return_a_list(self, video_results):
        for boxes in video_results:
            assert isinstance(boxes, list)

    def test_at_least_one_frame_detects_person(self, video_results):
        frames_with_person = sum(1 for boxes in video_results if len(boxes) > 0)
        assert frames_with_person > 0, "Expected at least one frame with a detected person"

    def test_all_bboxes_have_valid_coordinates(self, video_results):
        for boxes in video_results:
            for bb in boxes:
                assert bb.x1 < bb.x2, f"x1 >= x2: {bb}"
                assert bb.y1 < bb.y2, f"y1 >= y2: {bb}"

    def test_all_bboxes_have_valid_confidence(self, video_results):
        for boxes in video_results:
            for bb in boxes:
                assert 0.0 <= bb.confidence <= 1.0, f"Confidence out of range: {bb.confidence}"

    def test_detections_sorted_by_confidence_descending(self, video_results):
        for boxes in video_results:
            confidences = [bb.confidence for bb in boxes]
            assert confidences == sorted(confidences, reverse=True), (
                f"Detections not sorted by confidence: {confidences}"
            )
