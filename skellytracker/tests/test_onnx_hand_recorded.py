"""Standalone recorded-crop contracts; no Core imports or physical cameras.

W4_RECORDED_CROPS_DIR supplies lossless crop fixtures exported by the separately
owned application/reference harness. These tests exercise the actual shared stage
path, including MediaPipe's per-camera VIDEO task ownership.
"""

from contextlib import ExitStack
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.detectors.keypoint_detectors.mediapipe_onnx.hand_detector import (
    OnnxHandDetectorConfig,
)
from skellytracker.core.tracker.tracker import Tracker
from skellytracker.core.tracker.tracker_factory import build_tracker
from skellytracker.core.tracker.tracker_presets import mediapipe_hand_stage
from skellytracker.core.tracker.tracker_state import TrackerState


@pytest.fixture(params=("freemocap_test_data", "freemocap_sample_data"))
def recording_crops(request):
    directory = os.environ.get("W4_RECORDED_CROPS_DIR")
    if not directory:
        pytest.skip("Recorded crop fixtures not supplied")
    root = Path(directory)
    with np.load(root / f"{request.param}-paired.npz", allow_pickle=False) as arrays:
        cameras = tuple(arrays["camera_ids"].tolist())
        sides = tuple(arrays["hand_sides"].tolist())
        interval_ms = float(np.median(np.diff(arrays["source_seconds"])) * 1000)
    images = {}
    for side in sides:
        images[side] = {}
        for camera in cameras:
            image = cv2.imread(str(root / f"{request.param}-{side}-{camera}.png"))
            assert image is not None and image.size
            images[side][camera] = image
    return request.param, root, interval_ms, images


def hand_test_config(side, method):
    # Existing preset owns detector settings. The input fixture is already the
    # exact crop, so this laboratory stage must not perform a second crop.
    stage = mediapipe_hand_stage(side, parent_preset="rtmpose")
    stage = stage.model_copy(update={"object_detector": None, "bbox_smoothing": None})
    if method == "candidate":
        (baseline,) = stage.keypoint_detectors
        stage.keypoint_detectors = [
            OnnxHandDetectorConfig(
                assumed_handedness=side,
                min_hand_detection_confidence=baseline.min_hand_detection_confidence,
                min_hand_presence_confidence=baseline.min_hand_presence_confidence,
                model_directory=Path(os.environ["SKELLYTRACKER_HAND_CANDIDATE_DIR"]),
            )
        ]
    return TrackerConfig(stages=[stage])


def missing(points):
    return np.isnan(points.xyz).all() and not points.visibility.any()


@pytest.mark.parametrize("method", ("baseline", "candidate"))
@pytest.mark.parametrize("side", ("right", "left"))
def test_recorded_crops_camera_isolation_and_reacquisition(
    recording_crops, method, side
):
    dataset, artifact_root, interval_ms, all_images = recording_crops
    images = all_images[side]
    config = hand_test_config(side, method)
    stage_name = config.stages[0].name
    retained = []
    with ExitStack() as cleanup:
        tracker = build_tracker(
            config,
            batch_size=len(images),
            onnx_overrides={
                "execution_provider": "cpu",
                "fp16": False,
                "intra_op_num_threads": 1,
            },
        )
        cleanup.callback(tracker.close)
        singles = {
            camera: Tracker.create_with_shared_sessions(
                config=config, sessions=tracker.sessions
            )
            for camera in images
        }
        for single in singles.values():
            cleanup.callback(single.close)
        states = {}
        single_states = {camera: TrackerState() for camera in images}
        frame_number = 0
        cold_start_visible = {}

        def run(inputs):
            nonlocal states, frame_number
            timestamp = int(round(frame_number * interval_ms))
            observations, active_states = tracker.process_batch(
                inputs, frame_number, states, timestamp_ms=timestamp
            )
            states.update(active_states)
            assert observations.keys() == inputs.keys()
            points = {}
            for camera, image in inputs.items():
                independent, single_states[camera] = singles[camera].process_image(
                    image,
                    frame_number,
                    single_states[camera],
                    timestamp_ms=timestamp,
                )
                actual = observations[camera].stages[stage_name].keypoints
                expected = independent.stages[stage_name].keypoints
                assert actual.names == expected.names
                np.testing.assert_allclose(
                    actual.xyz, expected.xyz, rtol=1e-4, atol=1e-3, equal_nan=True
                )
                np.testing.assert_array_equal(actual.visibility, expected.visibility)
                retained.append((actual, actual.xyz.copy(), actual.visibility.copy()))
                points[camera] = actual
                cold_start_visible.setdefault(camera, bool(actual.n_valid))
            frame_number += 1
            return points

        # Each independent task sees the exact same per-camera sequence even when
        # group membership and iteration order change.
        for count in (1, len(images), max(1, len(images) - 1), 1):
            run(dict(list(reversed(list(images.items())))[:count]))
        before = run(images)
        pre_gap_visible = {
            camera: bool(points.n_valid) for camera, points in before.items()
        }
        # Fixtures were positive during a sequential recording. MediaPipe VIDEO
        # may depend on that history; cold detection is a measured outcome, not
        # guaranteed ground truth. The stateless candidate must reproduce it.
        if method == "candidate":
            assert all(pre_gap_visible.values())
        empty = run({camera: image[:0, :0] for camera, image in images.items()})
        assert all(missing(points) for points in empty.values())
        blank_count = max(1, int(np.ceil(500 / interval_ms)))
        for _ in range(blank_count):
            blank = run(
                {camera: np.zeros_like(image) for camera, image in images.items()}
            )
            assert all(missing(points) for points in blank.values())
        delay_frames = {camera: None for camera in images}
        for return_frame in range(max(2, int(np.ceil(1000 / interval_ms)))):
            returned = run(images)
            for camera, points in returned.items():
                if points.n_valid and delay_frames[camera] is None:
                    delay_frames[camera] = return_frame
            if all(delay is not None for delay in delay_frames.values()):
                break
        assert all(
            delay_frames[camera] is not None
            for camera, visible in pre_gap_visible.items()
            if visible
        ), delay_frames
        if method == "candidate":
            assert set(delay_frames.values()) == {0}
        tracker.reset_temporal_state()
        for single in singles.values():
            single.reset_temporal_state()
        states = {}
        single_states = {camera: TrackerState() for camera in images}
        frame_number = 0
        reset = run(images)
        reset_visible = {
            camera: bool(points.n_valid) for camera, points in reset.items()
        }
        assert reset_visible == cold_start_visible
    for points, values, scores in retained:
        np.testing.assert_array_equal(points.xyz, values)
        np.testing.assert_array_equal(points.visibility, scores)
    # Caller supplies an already checked/ignored evidence directory.
    report = dict(
        dataset=dataset,
        method=method,
        side=side,
        interval_ms=interval_ms,
        blank_duration_ms=blank_count * interval_ms,
        cold_start_visible=cold_start_visible,
        pre_gap_visible=pre_gap_visible,
        reset_visible=reset_visible,
        unreacquired_cameras=[
            camera for camera, delay in delay_frames.items() if delay is None
        ],
        recovery_window_ms=max(2, int(np.ceil(1000 / interval_ms))) * interval_ms,
        reacquisition_delay_frames=delay_frames,
        reacquisition_delay_ms={
            camera: None if delay is None else delay * interval_ms
            for camera, delay in delay_frames.items()
        },
        camera_isolation="matched independent per-camera task outputs",
        retained_results=len(retained),
    )
    (artifact_root / f"{dataset}-{side}-{method}-lifecycle.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
