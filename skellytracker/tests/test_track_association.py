from __future__ import annotations

import numpy as np
import pytest

from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.temporal_processing.multi_person_config import MultiPersonTrackingConfig
from skellytracker.core.temporal_processing.track_association import (
    associate,
    combined_cost_matrix,
    iou,
    iou_cost_matrix,
    keypoint_distance_cost_matrix,
)


def _box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _kpts(x: float, y: float, vis: float = 1.0) -> Keypoints:
    return Keypoints(
        names=("a", "b"),
        xyz=np.array([[x, y, 0.0], [x + 10.0, y + 10.0, 0.0]]),
        visibility=np.array([vis, vis]),
    )


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------

class TestIou:
    def test_identical_boxes(self):
        a = _box(0, 0, 100, 100)
        assert iou(a, a) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        a = _box(0, 0, 10, 10)
        b = _box(100, 100, 110, 110)
        assert iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = _box(0, 0, 10, 10)
        b = _box(5, 5, 15, 15)
        # intersection = 5x5 = 25, union = 100 + 100 - 25 = 175
        assert iou(a, b) == pytest.approx(25.0 / 175.0)


class TestCostMatrices:
    def test_iou_cost_matrix_shape_and_values(self):
        tracks = [_box(0, 0, 10, 10), None]
        dets = [_box(0, 0, 10, 10), _box(100, 100, 110, 110)]
        cost = iou_cost_matrix(tracks, dets)
        assert cost.shape == (2, 2)
        assert cost[0, 0] == pytest.approx(0.0)  # identical -> iou 1 -> cost 0
        assert cost[0, 1] == pytest.approx(1.0)  # disjoint -> iou 0 -> cost 1
        assert np.isinf(cost[1, :]).all()  # track with no bbox is fully gated

    def test_keypoint_distance_cost_matrix(self):
        tracks = [_kpts(0, 0), None]
        dets = [_kpts(0, 0), _kpts(1000, 1000)]
        cost = keypoint_distance_cost_matrix(tracks, dets)
        assert cost[0, 0] == pytest.approx(0.0)
        assert cost[0, 1] == pytest.approx(1.0)  # clipped to 1.0 for a huge displacement
        assert np.isinf(cost[1, :]).all()

    def test_combined_cost_falls_back_to_single_signal(self):
        # Track has a bbox but no keypoint history yet -> only IoU should count.
        tracks_bbox = [_box(0, 0, 10, 10)]
        tracks_kpts = [None]
        dets_bbox = [_box(0, 0, 10, 10)]
        dets_kpts = [_kpts(500, 500)]  # would look terrible on keypoints alone
        config = MultiPersonTrackingConfig()
        cost = combined_cost_matrix(tracks_bbox, tracks_kpts, dets_bbox, dets_kpts, config)
        assert cost[0, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# associate
# ---------------------------------------------------------------------------

class TestAssociate:
    def test_matches_two_tracks_to_two_of_three_detections(self):
        config = MultiPersonTrackingConfig()
        track_bboxes = [_box(0, 0, 10, 10), _box(200, 200, 210, 210)]
        track_kpts = [_kpts(0, 0), _kpts(200, 200)]
        det_bboxes = [
            _box(1, 1, 11, 11),      # close to track 0
            _box(201, 201, 211, 211),  # close to track 1
            _box(1000, 1000, 1010, 1010),  # far from everything -> new track
        ]
        det_kpts = [_kpts(1, 1), _kpts(201, 201), _kpts(1000, 1000)]

        result = associate(track_bboxes, track_kpts, det_bboxes, det_kpts, config)

        assert set(result.matches) == {(0, 0), (1, 1)}
        assert result.unmatched_tracks == []
        assert result.unmatched_detections == [2]

    def test_far_detection_is_gated_out_even_with_only_one_track(self):
        config = MultiPersonTrackingConfig(max_match_cost=0.5)
        track_bboxes = [_box(0, 0, 10, 10)]
        track_kpts = [_kpts(0, 0)]
        det_bboxes = [_box(1000, 1000, 1010, 1010)]
        det_kpts = [_kpts(1000, 1000)]

        result = associate(track_bboxes, track_kpts, det_bboxes, det_kpts, config)

        assert result.matches == []
        assert result.unmatched_tracks == [0]
        assert result.unmatched_detections == [0]

    def test_no_tracks_all_detections_unmatched(self):
        config = MultiPersonTrackingConfig()
        det_bboxes = [_box(0, 0, 10, 10)]
        det_kpts = [_kpts(0, 0)]

        result = associate([], [], det_bboxes, det_kpts, config)

        assert result.matches == []
        assert result.unmatched_tracks == []
        assert result.unmatched_detections == [0]

    def test_no_detections_all_tracks_unmatched(self):
        config = MultiPersonTrackingConfig()
        track_bboxes = [_box(0, 0, 10, 10)]
        track_kpts = [_kpts(0, 0)]

        result = associate(track_bboxes, track_kpts, [], [], config)

        assert result.matches == []
        assert result.unmatched_tracks == [0]
        assert result.unmatched_detections == []
