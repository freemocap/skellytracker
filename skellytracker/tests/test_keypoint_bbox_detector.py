from __future__ import annotations

import numpy as np
import pytest

from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.detectors.detector_base_classes import OBJECT_DETECTOR_REGISTRY
from skellytracker.core.detectors.object_detectors.keypoint_bbox import (
    KeypointBoundingBoxDetector,
    KeypointBoundingBoxDetectorConfig,
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


def _keypoints(named_xy: dict[str, tuple[float, float] | None], visibility: float = 1.0) -> Keypoints:
    names = tuple(named_xy)
    xyz = np.array(
        [
            [xy[0], xy[1], 0.0] if xy is not None else [np.nan, np.nan, np.nan]
            for xy in named_xy.values()
        ],
        dtype=np.float64,
    )
    vis = np.array(
        [visibility if xy is not None else 0.0 for xy in named_xy.values()],
        dtype=np.float64,
    )
    return Keypoints(names=names, xyz=xyz, visibility=vis)


class TestKeypointBoundingBoxDetectorCreate:
    def test_create_via_factory(self, cpu_session):
        config = KeypointBoundingBoxDetectorConfig(center_keypoint_names=("wrist",))
        detector = KeypointBoundingBoxDetector.create(config, cpu_session)
        assert isinstance(detector, KeypointBoundingBoxDetector)

    def test_registered_under_keypoint_bbox(self):
        assert OBJECT_DETECTOR_REGISTRY["keypoint_bbox"] is KeypointBoundingBoxDetector


class TestKeypointBoundingBoxDetectorDetect:
    def test_returns_no_boxes_without_parent_keypoints(self, image):
        detector = KeypointBoundingBoxDetector(
            config=KeypointBoundingBoxDetectorConfig(center_keypoint_names=("wrist",))
        )
        assert detector.detect(image, parent_keypoints=None) == []

    def test_centers_box_on_averaged_named_points(self, image):
        detector = KeypointBoundingBoxDetector(
            config=KeypointBoundingBoxDetectorConfig(center_keypoint_names=("index", "pinky"))
        )
        kpts = _keypoints({"wrist": (100.0, 100.0), "index": (120.0, 90.0), "pinky": (140.0, 90.0)})
        boxes = detector.detect(image, parent_keypoints=kpts)
        assert len(boxes) == 1
        cx, cy = boxes[0].center
        assert cx == pytest.approx(130.0)
        assert cy == pytest.approx(90.0)

    def test_sizes_box_from_largest_valid_scale_pair(self, image):
        detector = KeypointBoundingBoxDetector(
            config=KeypointBoundingBoxDetectorConfig(
                center_keypoint_names=("wrist",),
                scale_keypoint_pairs=(("wrist", "index"), ("wrist", "pinky")),
                scale_factor=2.0,
                min_box_size_px=1.0,
            )
        )
        kpts = _keypoints({"wrist": (0.0, 0.0), "index": (10.0, 0.0), "pinky": (30.0, 0.0)})
        boxes = detector.detect(image, parent_keypoints=kpts)
        assert len(boxes) == 1
        # largest pairwise distance (wrist->pinky = 30) * scale_factor 2.0 = 60
        assert boxes[0].width == pytest.approx(60.0)
        assert boxes[0].height == pytest.approx(60.0)

    def test_falls_back_to_min_box_size_when_no_scale_pairs_valid(self, image):
        detector = KeypointBoundingBoxDetector(
            config=KeypointBoundingBoxDetectorConfig(
                center_keypoint_names=("wrist",),
                scale_keypoint_pairs=(("wrist", "index"),),
                min_box_size_px=50.0,
            )
        )
        kpts = _keypoints({"wrist": (10.0, 10.0), "index": None})
        boxes = detector.detect(image, parent_keypoints=kpts)
        assert len(boxes) == 1
        assert boxes[0].width == pytest.approx(50.0)
        assert boxes[0].height == pytest.approx(50.0)

    def test_returns_no_boxes_when_all_center_points_invalid(self, image):
        detector = KeypointBoundingBoxDetector(
            config=KeypointBoundingBoxDetectorConfig(center_keypoint_names=("wrist",))
        )
        kpts = _keypoints({"wrist": None})
        assert detector.detect(image, parent_keypoints=kpts) == []

    def test_returns_no_boxes_when_center_points_below_min_visibility(self, image):
        detector = KeypointBoundingBoxDetector(
            config=KeypointBoundingBoxDetectorConfig(
                center_keypoint_names=("wrist",), min_visibility=0.5
            )
        )
        kpts = _keypoints({"wrist": (10.0, 10.0)}, visibility=0.1)
        assert detector.detect(image, parent_keypoints=kpts) == []

    def test_ignores_unnamed_points(self, image):
        detector = KeypointBoundingBoxDetector(
            config=KeypointBoundingBoxDetectorConfig(center_keypoint_names=("wrist", "missing_point"))
        )
        kpts = _keypoints({"wrist": (10.0, 20.0)})
        boxes = detector.detect(image, parent_keypoints=kpts)
        assert len(boxes) == 1
        assert boxes[0].center == pytest.approx((10.0, 20.0))
