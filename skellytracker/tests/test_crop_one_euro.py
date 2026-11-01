"""Crop filtering uses source time, isolated state and the executed pixel crop."""
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import ObjectDetector
from skellytracker.core.temporal_processing.bbox_smoothing import apply_bbox_one_euro
from skellytracker.core.temporal_processing.one_euro_array import OneEuroWorkspace, one_euro_step
from skellytracker.core.temporal_processing.temporal_processing_config import BBoxSmoothingConfig
from skellytracker.core.tracker.detection_stage import DetectionStage
from skellytracker.core.tracker.tracker_state import BBoxSmoothingState, StageState


def test_array_filter_matches_scalar_equations_and_reuses_workspace():
    rng = np.random.default_rng(101)
    values = rng.normal(size=(5, 7, 2))
    previous = rng.normal(size=values.shape)
    derivative = rng.normal(size=values.shape)
    original_previous = previous.copy()
    dt, cutoff, beta, d_cutoff = .1, 3., .02, 1.
    alpha_d = 1 / (1 + 1 / (2 * np.pi * d_cutoff * dt))
    expected_derivative = alpha_d * (values - previous) / dt + (1 - alpha_d) * derivative
    alpha = 1 / (1 + 1 / (2 * np.pi * (cutoff + beta * np.abs(expected_derivative)) * dt))
    expected_values = alpha * values + (1 - alpha) * previous
    work = OneEuroWorkspace.create(values.shape)
    out, speed = one_euro_step(values, previous, derivative, dt=dt,
        min_cutoff=cutoff, beta=beta, d_cutoff=d_cutoff, workspace=work)
    assert out is work.filtered_values and speed is work.derivative_values
    np.testing.assert_allclose(out, expected_values)
    np.testing.assert_allclose(speed, expected_derivative)
    np.testing.assert_array_equal(previous, original_previous)
    expected_next = one_euro_step(values, out.copy(), speed.copy(), dt=dt,
        min_cutoff=cutoff, beta=beta, d_cutoff=d_cutoff)[0]
    actual_next = one_euro_step(values, out, speed, dt=dt,
        min_cutoff=cutoff, beta=beta, d_cutoff=d_cutoff, workspace=work)[0]
    np.testing.assert_allclose(actual_next, expected_next)


@pytest.mark.parametrize('dt', [0., -1., np.nan, np.inf])
def test_invalid_elapsed_seconds_are_rejected(dt):
    with pytest.raises(ValueError):
        one_euro_step(np.ones(4), np.ones(4), np.zeros(4), dt=dt,
                      min_cutoff=3., beta=.02, d_cutoff=1.)


def test_missing_crop_and_long_pause_reacquire_without_old_position():
    config = BBoxSmoothingConfig()
    first = BoundingBox(x1=10, y1=10, x2=60, y2=60)
    moved = BoundingBox(x1=100, y1=100, x2=160, y2=160)
    _, state = apply_bbox_one_euro(first, BBoxSmoothingState(), config, None)
    smoothed, state = apply_bbox_one_euro(moved, state, config, 1/30)
    assert first.x1 < smoothed.x1 < moved.x1
    missing, state = apply_bbox_one_euro(None, state, config, 1/30)
    assert missing is None and state.corner_workspace is None
    reacquired, state = apply_bbox_one_euro(moved, state, config, 1/30)
    assert reacquired == moved
    reset, _ = apply_bbox_one_euro(first, state, config, .6)
    assert reset == first


def stage():
    detector = Mock(spec=ObjectDetector, session=SimpleNamespace(kind='cpu'))
    detector.detect.side_effect = lambda image, context, points: [BoundingBox(
        x1=float(image[0, 0, 0]), y1=20, x2=float(image[0, 0, 0])+50, y2=80)]
    return DetectionStage(name='crop', keypoint_detectors=[], object_detector=detector,
                          bbox_smoothing=BBoxSmoothingConfig())


def test_single_batch_parity_camera_isolation_and_stable_crop_storage():
    batch = stage()
    singles = {'a': stage(), 'b': stage()}
    batch_states, single_states = {}, {name: StageState() for name in singles}
    workspace_ids = None
    first_box = None
    for number, stamp in enumerate([0, 33, 100, 133, 166]):
        images = {'a': np.full((200, 250, 3), 30 + 20 * number, np.uint8),
                  'b': np.full((200, 250, 3), 150, np.uint8)}
        context = DetectionContext(frame_number=number, timestamp_ms=stamp)
        observations, batch_states = batch.run_batch(images, batch_states, context)
        for camera, image in images.items():
            single, single_states[camera] = singles[camera].run(image, single_states[camera], context=context)
            assert observations[camera].bounding_boxes == single.bounding_boxes
            box = single.bounding_boxes[0]
            assert box.to_crop(image).shape[:2] == (int(box.height), int(box.width))
        assert observations['b'].bounding_boxes[0].x1 == 150
        ids = tuple(id(batch_states[name].bbox_state.corner_workspace.filtered_values) for name in singles)
        if workspace_ids is None:
            workspace_ids = ids
            first_box = observations['a'].bounding_boxes[0]
        assert ids == workspace_ids and ids[0] != ids[1]
    assert first_box.x1 == 30  # Later workspace updates cannot mutate saved boxes.
    assert 30 < observations['a'].bounding_boxes[0].x1 < 110


def test_duplicate_backwards_or_missing_time_bypasses_stale_filter():
    current_stage = stage()
    state = StageState()
    for number, stamp in enumerate([100, 100, 50, None]):
        value = 20 + number * 30
        observation, state = current_stage.run(np.full((200, 250, 3), value, np.uint8), state,
            context=DetectionContext(frame_number=number, timestamp_ms=stamp))
        assert observation.bounding_boxes[0].x1 == value


def test_static_jitter_is_reduced_without_freezing_large_motion(record_property):
    rng = np.random.default_rng(29)
    config = BBoxSmoothingConfig()
    state = BBoxSmoothingState()
    raw_values, filtered_values = [], []
    for offset in rng.normal(0, 2, size=150):
        raw = BoundingBox(x1=100+offset, y1=100, x2=150+offset, y2=150)
        smooth, state = apply_bbox_one_euro(raw, state, config, 1/30)
        raw_values.append(raw.x1); filtered_values.append(smooth.x1)
    jitter_ratio = np.std(np.diff(filtered_values[30:])) / np.std(np.diff(raw_values[30:]))
    assert jitter_ratio < .6
    for frame in range(10):
        smooth, state = apply_bbox_one_euro(BoundingBox(x1=200,y1=100,x2=250,y2=150),state,config,1/30)
        if abs(smooth.x1 - 200) < 10:
            break
    assert frame < 5
    record_property('jitter_ratio', float(jitter_ratio))
    record_property('step_90_percent_seconds', (frame+1)/30)


@pytest.mark.parametrize('elapsed_ms', [33, 100, 200])
def test_stage_uses_elapsed_capture_time_in_filter_coefficients(elapsed_ms):
    current_stage = stage()
    current_stage.bbox_smoothing = BBoxSmoothingConfig(beta=0.0)
    first_image = np.full((200, 250, 3), 30, np.uint8)
    _, state = current_stage.run(first_image, StageState(), context=DetectionContext(timestamp_ms=1000))
    moved_image = np.full((200, 250, 3), 90, np.uint8)
    _, state = current_stage.run(moved_image, state,
        context=DetectionContext(frame_number=99, timestamp_ms=1000 + elapsed_ms))
    alpha = 1 / (1 + 1 / (2 * np.pi * 3.0 * elapsed_ms / 1000))
    assert state.bbox_state.corner_workspace.filtered_values[0, 0] == pytest.approx(30 + alpha * 60)
