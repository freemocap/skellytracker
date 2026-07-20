from __future__ import annotations

import numpy as np
import pytest

from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.object_detectors.precomputed import (
    PrecomputedObjectDetector,
    PrecomputedObjectDetectorConfig,
)
from skellytracker.core.sessions.cpu_session import CpuSession, CpuSessionConfig


@pytest.fixture(scope="module")
def cpu_session() -> CpuSession:
    session = CpuSession.create(CpuSessionConfig())
    yield session
    session.close()


@pytest.fixture
def image() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def two_boxes() -> list[BoundingBox]:
    return [
        BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0, confidence=0.9),
        BoundingBox(x1=300.0, y1=100.0, x2=500.0, y2=400.0, confidence=0.8),
    ]


class TestPrecomputedObjectDetectorCreate:
    def test_create_via_factory(self, cpu_session):
        config = PrecomputedObjectDetectorConfig()
        detector = PrecomputedObjectDetector.create(config, cpu_session)
        assert isinstance(detector, PrecomputedObjectDetector)

    def test_from_list_indexes_by_frame(self, two_boxes):
        frame0 = [two_boxes[0]]
        frame1 = [two_boxes[1]]
        detector = PrecomputedObjectDetector.from_list([frame0, frame1])
        assert detector.bboxes_by_frame[0] == frame0
        assert detector.bboxes_by_frame[1] == frame1


class TestPrecomputedObjectDetectorDetect:
    def test_returns_precomputed_boxes_for_known_frame(self, two_boxes, image):
        detector = PrecomputedObjectDetector.from_list([two_boxes])
        ctx = DetectionContext(frame_number=0)
        result = detector.detect(image, context=ctx)
        assert result == two_boxes

    def test_falls_back_to_full_image_for_unknown_frame(self, two_boxes, image):
        detector = PrecomputedObjectDetector.from_list([two_boxes])
        ctx = DetectionContext(frame_number=99)
        result = detector.detect(image, context=ctx)
        assert len(result) == 1
        bb = result[0]
        h, w = image.shape[:2]
        assert bb.x1 == pytest.approx(0.0)
        assert bb.y1 == pytest.approx(0.0)
        assert bb.x2 == pytest.approx(float(w))
        assert bb.y2 == pytest.approx(float(h))

    def test_falls_back_when_context_is_none(self, two_boxes, image):
        detector = PrecomputedObjectDetector.from_list([two_boxes])
        result = detector.detect(image, context=None)
        assert len(result) == 1
        h, w = image.shape[:2]
        assert result[0].x2 == pytest.approx(float(w))

    def test_empty_dict_always_returns_full_image(self, image):
        detector = PrecomputedObjectDetector()
        ctx = DetectionContext(frame_number=0)
        result = detector.detect(image, context=ctx)
        assert len(result) == 1

    def test_returned_boxes_are_valid_bounding_boxes(self, two_boxes, image):
        detector = PrecomputedObjectDetector.from_list([two_boxes])
        ctx = DetectionContext(frame_number=0)
        result = detector.detect(image, context=ctx)
        for bb in result:
            assert bb.x1 < bb.x2
            assert bb.y1 < bb.y2
            assert 0.0 <= bb.confidence <= 1.0

    def test_multiple_frames_return_correct_boxes(self, image):
        boxes_per_frame = [
            [BoundingBox(x1=float(i), y1=0.0, x2=float(i + 10), y2=10.0)]
            for i in range(5)
        ]
        detector = PrecomputedObjectDetector.from_list(boxes_per_frame)
        for frame_num, expected in enumerate(boxes_per_frame):
            ctx = DetectionContext(frame_number=frame_num)
            result = detector.detect(image, context=ctx)
            assert result == expected
