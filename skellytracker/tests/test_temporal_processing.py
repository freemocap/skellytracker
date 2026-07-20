from __future__ import annotations

import numpy as np
import pytest

from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.temporal_processing.bbox_policy import (
    BBoxAreaCollapseCheck,
    BBoxPolicy,
    KeypointsWithinBBoxRatioCheck,
    MaxFramesWithoutRedetectCheck,
    MinKeypointVisibilityCheck,
    predict_bbox_from_keypoints,
)
from skellytracker.core.temporal_processing.bbox_smoothing import apply_bbox_ema
from skellytracker.core.tracker.tracker_state import BBoxSmoothingState, StageState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stage(
    *,
    smooth_bbox: BoundingBox | None = None,
    last_detection_frame: int | None = None,
    last_keypoints: Keypoints | None = None,
    keypoint_tracked_bbox: BoundingBox | None = None,
) -> StageState:
    state = StageState()
    state.bbox_state.smooth_bbox = smooth_bbox
    state.bbox_state.last_detection_frame = last_detection_frame
    state.bbox_state.keypoint_tracked_bbox = keypoint_tracked_bbox
    state.last_keypoints = last_keypoints
    return state


def _kpts_grid(vis: float = 1.0) -> Keypoints:
    """Four keypoints spread across a 100×100 region starting at (50, 50)."""
    names = ("a", "b", "c", "d")
    xyz = np.array([
        [50.0, 50.0, 0.0],
        [150.0, 50.0, 0.0],
        [50.0, 150.0, 0.0],
        [150.0, 150.0, 0.0],
    ])
    visibility = np.full(4, vis)
    return Keypoints(names=names, xyz=xyz, visibility=visibility)


# ---------------------------------------------------------------------------
# predict_bbox_from_keypoints
# ---------------------------------------------------------------------------

class TestPredictBboxFromKeypoints:
    def test_basic_bbox(self):
        kpts = _kpts_grid()
        bb = predict_bbox_from_keypoints(kpts, expansion_ratio=0.0)
        assert bb is not None
        assert bb.x1 == pytest.approx(50.0)
        assert bb.y1 == pytest.approx(50.0)
        assert bb.x2 == pytest.approx(150.0)
        assert bb.y2 == pytest.approx(150.0)

    def test_expansion(self):
        kpts = _kpts_grid()
        bb = predict_bbox_from_keypoints(kpts, expansion_ratio=0.1)
        assert bb is not None
        assert bb.x1 < 50.0
        assert bb.y1 < 50.0
        assert bb.x2 > 150.0
        assert bb.y2 > 150.0

    def test_returns_none_all_nan(self):
        kpts = Keypoints.empty(("a", "b"))
        assert predict_bbox_from_keypoints(kpts, expansion_ratio=0.1) is None

    def test_min_visibility_filters(self):
        kpts = _kpts_grid(vis=0.1)
        result = predict_bbox_from_keypoints(kpts, expansion_ratio=0.0, min_visibility=0.5)
        assert result is None

    def test_single_point_returns_none(self):
        kpts = Keypoints(
            names=("p",),
            xyz=np.array([[100.0, 100.0, 0.0]]),
            visibility=np.array([1.0]),
        )
        assert predict_bbox_from_keypoints(kpts, expansion_ratio=0.0) is None


# ---------------------------------------------------------------------------
# MinKeypointVisibilityCheck
# ---------------------------------------------------------------------------

class TestMinKeypointVisibilityCheck:
    def test_fails_when_below_threshold(self):
        kpts = _kpts_grid(vis=0.1)
        state = _stage(last_keypoints=kpts)
        check = MinKeypointVisibilityCheck(threshold=0.5)
        assert check.fails(0, state)

    def test_passes_when_above_threshold(self):
        kpts = _kpts_grid(vis=0.9)
        state = _stage(last_keypoints=kpts)
        check = MinKeypointVisibilityCheck(threshold=0.5)
        assert not check.fails(0, state)

    def test_passes_when_no_keypoints(self):
        state = _stage(last_keypoints=None)
        check = MinKeypointVisibilityCheck(threshold=0.5)
        assert not check.fails(0, state)

    def test_passes_when_all_nan(self):
        kpts = Keypoints.empty(("a", "b"))
        state = _stage(last_keypoints=kpts)
        check = MinKeypointVisibilityCheck(threshold=0.5)
        assert not check.fails(0, state)


# ---------------------------------------------------------------------------
# KeypointsWithinBBoxRatioCheck
# ---------------------------------------------------------------------------

class TestKeypointsWithinBBoxRatioCheck:
    def test_all_inside_passes(self):
        kpts = _kpts_grid()
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=200.0, y2=200.0)
        state = _stage(smooth_bbox=bbox, last_keypoints=kpts)
        check = KeypointsWithinBBoxRatioCheck(threshold=1.0)
        assert not check.fails(0, state)

    def test_none_inside_fails(self):
        kpts = _kpts_grid()
        bbox = BoundingBox(x1=200.0, y1=200.0, x2=400.0, y2=400.0)
        state = _stage(smooth_bbox=bbox, last_keypoints=kpts)
        check = KeypointsWithinBBoxRatioCheck(threshold=0.5)
        assert check.fails(0, state)

    def test_passes_when_no_bbox(self):
        state = _stage(last_keypoints=_kpts_grid())
        check = KeypointsWithinBBoxRatioCheck(threshold=0.5)
        assert not check.fails(0, state)

    def test_passes_when_no_keypoints(self):
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=200.0, y2=200.0)
        state = _stage(smooth_bbox=bbox)
        check = KeypointsWithinBBoxRatioCheck(threshold=0.5)
        assert not check.fails(0, state)


# ---------------------------------------------------------------------------
# MaxFramesWithoutRedetectCheck
# ---------------------------------------------------------------------------

class TestMaxFramesWithoutRedetectCheck:
    def test_fails_when_no_detection(self):
        state = _stage(last_detection_frame=None)
        check = MaxFramesWithoutRedetectCheck(n_frames=10)
        assert check.fails(5, state)

    def test_fails_when_gap_exceeds_limit(self):
        state = _stage(last_detection_frame=0)
        check = MaxFramesWithoutRedetectCheck(n_frames=10)
        assert check.fails(10, state)

    def test_passes_when_recently_detected(self):
        state = _stage(last_detection_frame=5)
        check = MaxFramesWithoutRedetectCheck(n_frames=10)
        assert not check.fails(10, state)

    def test_boundary_at_limit(self):
        state = _stage(last_detection_frame=0)
        check = MaxFramesWithoutRedetectCheck(n_frames=10)
        assert check.fails(10, state)
        assert not check.fails(9, state)


# ---------------------------------------------------------------------------
# BBoxAreaCollapseCheck
# ---------------------------------------------------------------------------

class TestBBoxAreaCollapseCheck:
    def test_passes_when_no_state(self):
        check = BBoxAreaCollapseCheck(min_area_ratio=0.25, expansion_ratio=0.05, min_visibility=0.0)
        assert not check.fails(0, _stage())

    def test_fails_when_keypoints_outside_bbox(self):
        # Keypoints cluster at (500–600, 500–600) → predicted area ≈ 110×110 = 12100
        # Smooth bbox is 300×300 = 90000; 12100/90000 ≈ 0.13 < min_area_ratio=0.25 → should fail
        kpts = Keypoints(
            names=("a", "b"),
            xyz=np.array([[500.0, 500.0, 0.0], [600.0, 600.0, 0.0]]),
            visibility=np.array([1.0, 1.0]),
        )
        large_bbox = BoundingBox(x1=0.0, y1=0.0, x2=300.0, y2=300.0)
        state = _stage(smooth_bbox=large_bbox, last_keypoints=kpts)
        check = BBoxAreaCollapseCheck(min_area_ratio=0.25, expansion_ratio=0.05, min_visibility=0.0)
        assert check.fails(0, state)

    def test_passes_when_keypoints_span_large_region(self):
        kpts = _kpts_grid()
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=200.0, y2=200.0)
        state = _stage(smooth_bbox=bbox, last_keypoints=kpts)
        check = BBoxAreaCollapseCheck(min_area_ratio=0.25, expansion_ratio=0.05, min_visibility=0.0)
        assert not check.fails(0, state)


# ---------------------------------------------------------------------------
# BBoxPolicy
# ---------------------------------------------------------------------------

class TestBBoxPolicy:
    def test_redetects_on_first_frame(self):
        policy = BBoxPolicy(redetect_interval=5)
        state = _stage()
        assert policy.should_redetect(0, state)

    def test_skips_within_interval(self):
        policy = BBoxPolicy(redetect_interval=5)
        state = _stage(last_detection_frame=0)
        assert not policy.should_redetect(3, state)

    def test_redetects_at_interval(self):
        policy = BBoxPolicy(redetect_interval=5)
        state = _stage(last_detection_frame=0)
        assert policy.should_redetect(5, state)

    def test_fitness_check_triggers_redetect(self):
        policy = BBoxPolicy(
            redetect_interval=100,
            fitness_checks=[MaxFramesWithoutRedetectCheck(n_frames=2)],
        )
        state = _stage(last_detection_frame=0)
        assert policy.should_redetect(5, state)

    def test_predict_bbox_falls_back_to_smooth(self):
        bbox = BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0)
        state = _stage(smooth_bbox=bbox)
        policy = BBoxPolicy()
        result = policy.predict_bbox(state)
        assert result is bbox

    def test_predict_bbox_uses_keypoints_when_configured(self):
        tracked = predict_bbox_from_keypoints(_kpts_grid(), expansion_ratio=0.0)
        state = _stage(keypoint_tracked_bbox=tracked)
        policy = BBoxPolicy(
            keypoint_bbox_expansion=0.0,
            min_shrink_ratio_per_frame=None,
            min_detected_bbox_ratio=None,
            min_bbox_size_px=0.0,
        )
        result = policy.predict_bbox(state)
        assert result is not None
        assert result.x1 == pytest.approx(50.0)

    def test_predict_bbox_returns_none_without_state(self):
        state = _stage()
        policy = BBoxPolicy()
        assert policy.predict_bbox(state) is None


# ---------------------------------------------------------------------------
# apply_bbox_ema
# ---------------------------------------------------------------------------

class TestApplyBboxEma:
    def test_first_frame_passes_through(self):
        raw = BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0)
        state = BBoxSmoothingState()
        smooth, new_state = apply_bbox_ema(raw, state, alpha=0.5)
        assert smooth.x1 == pytest.approx(10.0)
        assert new_state.smooth_bbox is smooth

    def test_ema_blends_toward_raw(self):
        prev = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        raw = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=200.0)
        state = BBoxSmoothingState(smooth_bbox=prev)
        smooth, _ = apply_bbox_ema(raw, state, alpha=0.5)
        prev_cx, prev_cy = prev.center
        raw_cx, raw_cy = raw.center
        expected_cx = 0.5 * prev_cx + 0.5 * raw_cx
        assert smooth.center[0] == pytest.approx(expected_cx)

    def test_alpha_zero_equals_raw(self):
        prev = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        raw = BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=150.0)
        state = BBoxSmoothingState(smooth_bbox=prev)
        smooth, _ = apply_bbox_ema(raw, state, alpha=0.0)
        assert smooth.center == pytest.approx(raw.center)
        assert smooth.width == pytest.approx(raw.width)

    def test_state_is_updated(self):
        raw = BoundingBox(x1=1.0, y1=1.0, x2=2.0, y2=2.0)
        state = BBoxSmoothingState()
        _, new_state = apply_bbox_ema(raw, state, alpha=0.5)
        assert new_state.smooth_bbox is not None
        assert state.smooth_bbox is None
