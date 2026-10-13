"""Batch mapping must preserve scalar results and missing-observation masks."""

from pathlib import Path

import numpy as np
import pytest

from skellytracker.core.io.composite_tracker_mapping import CompositeTrackerMapping
from skellytracker.core.io.tracker_mapping import TrackerMapping
from skellytracker.core.io.tracker_mapping_batch import map_tracker_batch

MAPPINGS = sorted(
    Path("skellytracker/core/detectors/keypoint_detectors").rglob(
        "*_to_standard_human_mapping.yaml"
    )
)


def compare(mapping, names, values):
    from skellytracker.core.io.prepared_tracker_mapping import PreparedTrackerMapping

    prepared = PreparedTrackerMapping.prepare(mapping, tracker_names=names)
    batch = prepared.apply(tracker_names=names, positions=values)
    original = map_tracker_batch(mapping, tracker_names=names, positions=values)
    assert batch.landmark_names == original.landmark_names
    np.testing.assert_array_equal(batch.present, original.present)
    np.testing.assert_allclose(
        batch.positions, original.positions, rtol=1e-12, atol=1e-10
    )

    for frame, positions in enumerate(values):
        expected = mapping.apply(
            {
                name: value
                for name, value in zip(names, positions, strict=True)
                if np.isfinite(value).all()
            }
        )
        present = {
            name
            for column, name in enumerate(batch.landmark_names)
            if batch.present[frame, column]
        }
        assert present == set(expected)
        for name, value in expected.items():
            np.testing.assert_allclose(
                batch.positions[frame, batch.landmark_names.index(name)],
                value,
                rtol=1e-12,
                atol=1e-10,
            )
    return batch


@pytest.mark.parametrize("path", MAPPINGS, ids=lambda p: p.stem)
@pytest.mark.parametrize("missing", [False, True])
def test_authored_mappings(path, missing):
    mapping = TrackerMapping.from_yaml(path)
    names = tuple(dict.fromkeys(mapping.tracker_names))
    values = np.random.default_rng(72).normal(size=(100, len(names), 3)) * 100
    if missing:
        values[::7] = np.nan
        values[1::3, ::5] = np.nan
        values[2::11] = 0  # degenerate axes and zero reference lengths
    compare(mapping, names, values)


def test_affine_prefix_and_composite():
    mapping = CompositeTrackerMapping(
        (
            TrackerMapping(
                entries={"a": "x", "b": ["x", "y"], "c": {"x": 2.0, "y": -1.0}},
                prefix="body:",
            ),
            TrackerMapping(
                entries={},
                known_tracker_keypoints={"board:z"},
                prefix="board:",
                passthrough_keypoints_as_landmarks=True,
            ),
        )
    )
    values = np.arange(90, dtype=float).reshape(10, 3, 3)
    values[1, 0, 1] = np.nan
    compare(mapping, ("body:x", "body:y", "board:z"), values)


def test_invalid_axes_and_empty_recording():
    mapping = TrackerMapping(entries={"a": "x"})
    with pytest.raises(ValueError):
        map_tracker_batch(
            mapping, tracker_names=("x", "x"), positions=np.zeros((2, 2, 3))
        )
    with pytest.raises(ValueError):
        map_tracker_batch(mapping, tracker_names=("x",), positions=np.zeros((2, 3)))
    assert map_tracker_batch(
        mapping, tracker_names=("x",), positions=np.empty((0, 1, 3))
    ).positions.shape == (0, 1, 3)
    assert map_tracker_batch(
        mapping, tracker_names=(), positions=np.empty((2, 0, 3))
    ).positions.shape == (2, 0, 3)


def test_prepared_mapping_reuse_and_invalidation():
    from skellytracker.core.io.prepared_tracker_mapping import PreparedTrackerMapping

    mapping = TrackerMapping(entries={"a": "x", "b": ["x", "y"]})
    names = ("x", "y")
    plan = PreparedTrackerMapping.prepare(mapping, tracker_names=names)
    assert (
        plan.fingerprint
        == PreparedTrackerMapping.prepare(mapping, tracker_names=names).fingerprint
    )
    changed = PreparedTrackerMapping.prepare(
        TrackerMapping(entries={"a": "y"}), tracker_names=names
    )
    assert changed.fingerprint != plan.fingerprint
    with pytest.raises(ValueError, match="axis changed"):
        plan.apply(tracker_names=("y", "x"), positions=np.zeros((1, 2, 3)))

    def unexpected():
        raise AssertionError("Definitions must not be read during frame processing")

    mapping.mapping_snapshots = unexpected
    first = plan.apply(tracker_names=names, positions=np.ones((1, 2, 3)))
    preserved = first.positions.copy()
    missing = np.ones((1, 2, 3))
    missing[:, 1] = np.nan
    second = plan.apply(tracker_names=names, positions=missing)
    np.testing.assert_array_equal(second.present, [[True, False]])
    np.testing.assert_array_equal(first.positions, preserved)
    third = plan.apply(tracker_names=names, positions=np.ones((1, 2, 3)) * 4)
    np.testing.assert_array_equal(third.positions, np.ones((1, 2, 3)) * 4)
    assert not np.shares_memory(first.positions, third.positions)
    with pytest.raises(ValueError):
        plan.components[0].affine[0].sources[0, 0] = 1


def test_prepared_empty_frames():
    from skellytracker.core.io.prepared_tracker_mapping import PreparedTrackerMapping

    mapping = TrackerMapping.from_yaml(MAPPINGS[0])
    names = tuple(dict.fromkeys(mapping.tracker_names))
    plan = PreparedTrackerMapping.prepare(mapping, tracker_names=names)
    output = plan.apply(tracker_names=names, positions=np.empty((0, len(names), 3)))
    assert output.positions.shape == (0, len(plan.landmark_names), 3)


@pytest.mark.parametrize("detector", ["rtmpose", "mediapipe"])
def test_wholebody_schema_and_composed_mapping(detector):
    from skellytracker.core.detectors.keypoint_detectors._schema_loader import (
        load_point_names,
    )

    root = Path("skellytracker/core/detectors/keypoint_detectors")
    hand_folder = "hand" if detector == "rtmpose" else "hands"
    mapping = CompositeTrackerMapping(
        (
            TrackerMapping.from_yaml(
                root
                / detector
                / "body"
                / f"{detector}_body_to_standard_human_mapping.yaml"
            ),
            TrackerMapping.from_yaml(
                root
                / detector
                / hand_folder
                / f"{detector}_hand_to_standard_human_mapping.yaml"
            ),
        )
    )
    if detector == "rtmpose":
        names = tuple(
            sorted(
                load_point_names(root / detector / "wholebody/rtmpose_wholebody.yaml")
            )
        )
    else:
        names_set = set(load_point_names(root / detector / "body/mediapipe_body.yaml"))
        names_set.update(
            load_point_names(root / detector / "face/mediapipe_face_contour.yaml")
        )
        for side in ("left", "right"):
            names_set.update(
                f"{side}_hand_{name}"
                for name in load_point_names(
                    root / detector / "hands/mediapipe_hand.yaml"
                )
            )
        names = tuple(sorted(names_set))
    values = np.random.default_rng(8).normal(size=(12, len(names), 3)) * 100
    values[1, ::3] = np.nan
    values[2] = np.nan
    compare(mapping, names, values)


@pytest.mark.parametrize("count", [36, 92, 478])
def test_dense_passthrough_markers(count):
    names = tuple(f"point_{index}" for index in range(count))
    mapping = TrackerMapping(
        entries={},
        known_tracker_keypoints=set(names),
        passthrough_keypoints_as_landmarks=True,
    )
    values = np.random.default_rng(9).normal(size=(4, count, 3))
    values[1, ::2] = np.nan
    compare(mapping, names, values)


def test_prepared_frame_interface():
    from skellytracker.core.io.prepared_tracker_mapping import PreparedTrackerMapping

    mapping = TrackerMapping(entries={"a": "x", "b": ["x", "y"]})
    plan = PreparedTrackerMapping.prepare(mapping, tracker_names=("x", "y"))
    for points in ({}, {"x": np.ones(3)}, {"y": np.zeros(3), "x": np.ones(3)}):
        actual = plan.apply_frame(tracker_positions=points)
        expected = mapping.apply(points)
        assert actual.keys() == expected.keys()
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])
    with pytest.raises(ValueError, match="axis changed"):
        plan.apply_frame(tracker_positions={"z": np.ones(3)})
    with pytest.raises(ValueError, match="three coordinates"):
        plan.apply_frame(tracker_positions={"x": np.ones(2)})


def test_configuration_replacement():
    from skellytracker.core.io.prepared_tracker_mapping import PreparedTrackerMapping

    original = TrackerMapping(entries={"a": "x"})
    old = PreparedTrackerMapping.prepare(original, tracker_names=("x", "y"))
    changed = PreparedTrackerMapping.prepare(
        TrackerMapping(entries={"a": "y"}), tracker_names=("x", "y")
    )
    points = {"x": np.ones(3), "y": np.ones(3) * 7}
    np.testing.assert_array_equal(
        old.apply_frame(tracker_positions=points)["a"], np.ones(3)
    )
    np.testing.assert_array_equal(
        changed.apply_frame(tracker_positions=points)["a"], np.ones(3) * 7
    )
    assert old.fingerprint != changed.fingerprint
    reordered = PreparedTrackerMapping.prepare(original, tracker_names=("y", "x"))
    assert reordered.fingerprint != old.fingerprint
    np.testing.assert_array_equal(
        reordered.apply_frame(tracker_positions=points)["a"], np.ones(3)
    )
    boards = [
        PreparedTrackerMapping.prepare(
            TrackerMapping(
                entries={},
                known_tracker_keypoints={f"p{i}" for i in range(n)},
                passthrough_keypoints_as_landmarks=True,
            ),
            tracker_names=tuple(f"p{i}" for i in range(n)),
        )
        for n in (36, 92)
    ]
    assert boards[0].fingerprint != boards[1].fingerprint
    assert len(boards[1].landmark_names) == 92
    detector_plans = [
        PreparedTrackerMapping.prepare(
            TrackerMapping.from_yaml(path),
            tracker_names=tuple(
                dict.fromkeys(TrackerMapping.from_yaml(path).tracker_names)
            ),
        )
        for path in MAPPINGS
    ]
    assert len({plan.fingerprint for plan in detector_plans}) == len(MAPPINGS)


@pytest.mark.parametrize("path", MAPPINGS, ids=lambda p: p.stem)
def test_prepared_quality(path):
    from skellytracker.core.io.prepared_tracker_mapping import PreparedTrackerMapping

    mapping = TrackerMapping.from_yaml(path)
    names = tuple(dict.fromkeys(mapping.tracker_names))
    plan = PreparedTrackerMapping.prepare(mapping, tracker_names=names)
    rng = np.random.default_rng(77)
    points = rng.normal(size=(20, len(names), 3)) * 100
    points[::3, ::5] = np.nan
    scores = rng.uniform(size=points.shape[:2])
    output = plan.apply_with_quality(
        tracker_names=names, positions=points, quality=scores
    )
    for row, frame in enumerate(points):
        positions = {
            name: value
            for name, value in zip(names, frame, strict=True)
            if np.isfinite(value).all()
        }
        qualities = {
            name: scores[row, index]
            for index, name in enumerate(names)
            if name in positions
        }
        expected = mapping.apply_with_quality(
            tracker_positions=positions, tracker_quality=qualities
        )
        actual = {
            name: output.quality[row, index]
            for index, name in enumerate(output.landmark_names)
            if output.present[row, index]
        }
        assert actual == expected.quality
    with pytest.raises(ValueError, match="Quality must match"):
        plan.apply_with_quality(
            tracker_names=names, positions=points, quality=scores[:1]
        )
    with pytest.raises(ValueError, match="source quality"):
        plan.apply_with_quality(
            tracker_names=names, positions=points, quality=np.full_like(scores, 2)
        )


@pytest.mark.parametrize("path", MAPPINGS, ids=lambda p: p.stem)
def test_prepared_chunks_and_concurrent_calls(path):
    from concurrent.futures import ThreadPoolExecutor

    from skellytracker.core.io.prepared_tracker_mapping import PreparedTrackerMapping

    mapping = TrackerMapping.from_yaml(path)
    names = tuple(dict.fromkeys(mapping.tracker_names))
    plan = PreparedTrackerMapping.prepare(mapping, tracker_names=names)
    points = np.random.default_rng(55).normal(size=(17, len(names), 3))
    points[::4, ::3] = np.nan
    scores = np.full(points.shape[:2], 0.8)
    expected = plan.apply_with_quality(
        tracker_names=names, positions=points, quality=scores
    )

    def run(bounds):
        start, stop = bounds
        return plan.apply_with_quality(
            tracker_names=names,
            positions=points[start:stop],
            quality=scores[start:stop],
        )

    for size in (1, 3, 8):
        bounds = [
            (start, min(start + size, len(points)))
            for start in range(0, len(points), size)
        ]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, bounds))
        np.testing.assert_allclose(
            np.concatenate([r.positions for r in results]),
            expected.positions,
            rtol=1e-12,
            atol=1e-10,
        )
        np.testing.assert_array_equal(
            np.concatenate([r.present for r in results]), expected.present
        )
        np.testing.assert_array_equal(
            np.concatenate([r.quality for r in results]), expected.quality
        )
        assert not np.shares_memory(results[0].positions, expected.positions)


def test_quality_composite_and_passthrough():
    from skellytracker.core.io.prepared_tracker_mapping import PreparedTrackerMapping

    mapping = CompositeTrackerMapping(
        (
            TrackerMapping(entries={"mid": ["x", "y"]}),
            TrackerMapping(
                entries={},
                known_tracker_keypoints={"z"},
                passthrough_keypoints_as_landmarks=True,
            ),
        )
    )
    plan = PreparedTrackerMapping.prepare(mapping, tracker_names=("x", "y", "z"))
    points = np.ones((2, 3, 3))
    points[1, 0] = np.nan
    scores = np.array([[0.5, 0.2, 0.9], [np.nan, 0.2, 0.9]])
    result = plan.apply_with_quality(
        tracker_names=plan.tracker_names, positions=points, quality=scores
    )
    np.testing.assert_array_equal(result.quality, [[0.2, 0.9], [np.nan, 0.9]])
