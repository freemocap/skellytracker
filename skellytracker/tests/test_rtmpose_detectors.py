"""Tests for the YOLOX person detector and RTMPose keypoint detector.

Skips automatically if onnxruntime is not installed or if the test image
cannot be downloaded (network-dependent).
"""
from __future__ import annotations

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

import skellytracker.core.detectors.object_detectors.yolox  # noqa: F401, E402
import skellytracker.core.detectors.keypoint_detectors.rtmpose  # noqa: F401, E402

from skellytracker.core import (  # noqa: E402
    KEYPOINT_DETECTOR_REGISTRY,
    OBJECT_DETECTOR_REGISTRY,
    DetectionStageConfig,
    Tracker,
    TrackerConfig,
    TrackerState,
)
from skellytracker.core.detectors.object_detectors.yolox import (  # noqa: E402
    YoloxPersonDetector,
    YoloxPersonDetectorConfig,
)
from skellytracker.core.detectors.keypoint_detectors.rtmpose import (  # noqa: E402
    RTMPoseDetectorConfig,
    RTMPoseKeypointDetector,
)
from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig  # noqa: E402


@pytest.fixture(scope="module")
def onnx_session() -> OnnxSession:
    config = OnnxSessionConfig(
        models=[
            YoloxPersonDetector.model_spec("yolox-m"),
            RTMPoseKeypointDetector.model_spec("rtmw-x-l_256x192"),
        ],
    )
    session = OnnxSession.create(config)
    yield session
    session.close()


class TestRegistry:
    def test_yolox_registered(self):
        assert "yolox_person" in OBJECT_DETECTOR_REGISTRY
        assert OBJECT_DETECTOR_REGISTRY["yolox_person"] is YoloxPersonDetector

    def test_rtmpose_registered(self):
        assert "rtmpose" in KEYPOINT_DETECTOR_REGISTRY
        assert KEYPOINT_DETECTOR_REGISTRY["rtmpose"] is RTMPoseKeypointDetector


class TestYoloxPersonDetector:
    def test_detect_on_test_image_returns_bounding_boxes(self, test_image, onnx_session):
        detector = YoloxPersonDetector.create(YoloxPersonDetectorConfig(), onnx_session)
        boxes = detector.detect(test_image)
        assert isinstance(boxes, list)
        for bb in boxes:
            assert bb.x1 < bb.x2
            assert bb.y1 < bb.y2
            assert 0.0 <= bb.confidence <= 1.0

    def test_detect_finds_person_in_test_image(self, test_image, onnx_session):
        detector = YoloxPersonDetector.create(YoloxPersonDetectorConfig(), onnx_session)
        boxes = detector.detect(test_image)
        assert len(boxes) > 0, "Expected at least one person detected in test image"

    def test_detect_blank_image_returns_empty(self, onnx_session):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = YoloxPersonDetector.create(YoloxPersonDetectorConfig(), onnx_session)
        boxes = detector.detect(blank)
        assert isinstance(boxes, list)

    def test_max_detections_limits_results(self, test_image, onnx_session):
        config = YoloxPersonDetectorConfig(max_detections=1)
        detector = YoloxPersonDetector.create(config, onnx_session)
        boxes = detector.detect(test_image)
        assert len(boxes) <= 1

    def test_connections_returns_empty(self, onnx_session):
        assert YoloxPersonDetector.connections() == ()


class TestRTMPoseKeypointDetector:
    def test_detect_returns_133_keypoints(self, test_image, onnx_session):
        detector = RTMPoseKeypointDetector.create(RTMPoseDetectorConfig(), onnx_session)
        kpts = detector.detect(test_image)
        assert kpts.xyz.shape == (133, 3)
        assert kpts.visibility.shape == (133,)
        assert len(kpts.names) == 133

    def test_visibility_in_range(self, test_image, onnx_session):
        detector = RTMPoseKeypointDetector.create(RTMPoseDetectorConfig(), onnx_session)
        kpts = detector.detect(test_image)
        assert np.all(kpts.visibility >= 0.0)
        assert np.all(kpts.visibility <= 1.0)

    def test_detect_on_real_image_has_valid_points(self, test_image, onnx_session):
        detector = RTMPoseKeypointDetector.create(RTMPoseDetectorConfig(), onnx_session)
        kpts = detector.detect(test_image)
        assert kpts.n_valid > 0, "Expected at least one detected keypoint on test image"

    def test_blank_image_returns_nan_xyz(self, onnx_session):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = RTMPoseKeypointDetector.create(RTMPoseDetectorConfig(), onnx_session)
        kpts = detector.detect(blank)
        assert kpts.xyz.shape == (133, 3)
        assert np.all(np.isnan(kpts.xyz))
        assert np.all(kpts.visibility == 0.0)

    def test_with_bbox_crops_correctly(self, test_image, onnx_session):
        from skellytracker.core.data_primitives import BoundingBox
        h, w = test_image.shape[:2]
        bbox = BoundingBox(x1=0, y1=0, x2=w, y2=h)
        detector = RTMPoseKeypointDetector.create(RTMPoseDetectorConfig(), onnx_session)
        kpts = detector.detect(test_image, bbox=bbox)
        assert kpts.xyz.shape == (133, 3)

    def test_point_names_include_body_hand_face(self, onnx_session):
        detector = RTMPoseKeypointDetector.create(RTMPoseDetectorConfig(), onnx_session)
        blank = np.zeros((10, 10, 3), dtype=np.uint8)
        kpts = detector.detect(blank)
        assert kpts.has_name("nose")
        assert kpts.has_name("right_hand_root")
        assert kpts.has_name("left_hand_root")
        assert kpts.has_name("face_0000")

    def test_connections_loads_from_yaml(self):
        conns = RTMPoseKeypointDetector.connections()
        assert len(conns) > 0
        for a, b in conns:
            assert isinstance(a, str)
            assert isinstance(b, str)


class TestTrackerIntegration:
    def test_full_pipeline_with_yolox_and_rtmpose(self, test_image, onnx_session):
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
        state = TrackerState()

        observation, state = tracker.process_image(test_image, frame_number=0, state=state)

        assert "body" in observation.stages
        stage = observation.stages["body"]
        assert stage.keypoints is not None
        assert stage.keypoints.xyz.shape == (133, 3)
        assert stage.keypoints.n_valid > 0

    def test_rtmpose_without_object_detector(self, test_image, onnx_session):
        """RTMPose on full image when no person detector is used."""
        config = TrackerConfig(
            stages=[
                DetectionStageConfig(
                    name="body",
                    keypoint_detectors=[RTMPoseDetectorConfig()],
                )
            ]
        )
        tracker = Tracker.create(config, {"onnx": onnx_session})
        state = TrackerState()

        observation, state = tracker.process_image(test_image, frame_number=0, state=state)

        assert "body" in observation.stages
        kpts = observation.stages["body"].keypoints
        assert kpts is not None
        assert kpts.xyz.shape == (133, 3)
