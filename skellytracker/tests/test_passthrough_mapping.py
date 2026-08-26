"""Pass-through mapping: for an object whose markers ARE its landmarks.

A charuco board has no anatomy to map onto, so the mapping is one flag rather than one
line per marker. These tests pin the two things that make it usable rather than merely
short: it produces exactly what the tracker emits, and it can say so UP FRONT — callers
that decide which landmarks are measured, and therefore which segments may set the model
scale, cannot wait until the first frame to find out.
"""

from __future__ import annotations

import numpy as np
import pytest

from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_board_definition import (
    CharucoBoardDefinition,
)
from skellytracker.core.io.mapping_paths import (
    CHARUCO_BOARD_MAPPING,
    MEDIAPIPE_BODY_MAPPING,
)
from skellytracker.core.io.tracker_mapping import TrackerMapping


def _board_mapping() -> tuple[CharucoBoardDefinition, TrackerMapping]:
    board = CharucoBoardDefinition.create_letter_size_5x3()
    return board, TrackerMapping.from_yaml(
        CHARUCO_BOARD_MAPPING, known_tracker_keypoints=set(board.all_point_names)
    )


def test_the_shipped_board_mapping_is_a_passthrough() -> None:
    _, mapping = _board_mapping()
    assert mapping.is_passthrough


def test_every_keypoint_becomes_a_landmark_of_the_same_name() -> None:
    board, mapping = _board_mapping()
    observed = {
        name: np.array([float(index), 1.0, 2.0])
        for index, name in enumerate(board.all_point_names)
    }

    produced = mapping.apply(tracker_positions=observed)

    assert set(produced) == set(board.all_point_names)
    for name, position in observed.items():
        np.testing.assert_array_equal(produced[name], position)


def test_a_partially_seen_board_produces_only_what_was_seen() -> None:
    """Occlusion is data — the mapping does not invent the corners off-frame."""
    board, mapping = _board_mapping()
    seen = dict(
        zip(board.charuco_corner_names[:3], [np.zeros(3), np.ones(3), np.full(3, 2.0)])
    )

    produced = mapping.apply(tracker_positions=seen)

    assert set(produced) == set(board.charuco_corner_names[:3])


def test_a_name_the_board_does_not_have_is_dropped() -> None:
    """Better than becoming a landmark nothing downstream has ever heard of."""
    board, mapping = _board_mapping()
    produced = mapping.apply(
        tracker_positions={
            board.charuco_corner_names[0]: np.zeros(3),
            "SomeOtherDetectorsPoint": np.ones(3),
        }
    )
    assert set(produced) == {board.charuco_corner_names[0]}


def test_a_passthrough_measures_everything_it_produces() -> None:
    """There is nothing constructed to exclude, so every landmark may set the scale."""
    board, mapping = _board_mapping()
    assert mapping.directly_measured_landmark_names == frozenset(board.all_point_names)


def test_a_passthrough_without_known_keypoints_is_refused() -> None:
    """It could only answer 'what do you produce?' at apply time, which is too late."""
    with pytest.raises(ValueError, match="known_tracker_keypoints"):
        TrackerMapping(entries={}, passthrough_keypoints_as_landmarks=True)


def test_a_mapping_cannot_be_both_passthrough_and_entries() -> None:
    with pytest.raises(ValueError, match="cannot also list entries"):
        TrackerMapping(
            entries={"a_landmark": "a_keypoint"},
            known_tracker_keypoints={"a_keypoint"},
            passthrough_keypoints_as_landmarks=True,
        )


def test_an_ordinary_mapping_is_unaffected() -> None:
    """The flag is opt-in; the human mappings keep constructing what they construct."""
    mapping = TrackerMapping.from_yaml(MEDIAPIPE_BODY_MAPPING)
    assert not mapping.is_passthrough
    assert "left_knee" in mapping.directly_measured_landmark_names
    assert "chest_center" not in mapping.directly_measured_landmark_names


def test_the_board_mapping_covers_any_board_size() -> None:
    """The flag is the whole file, so a 7x5 board needs no second mapping."""
    board = CharucoBoardDefinition.create_test_data_7x5()
    mapping = TrackerMapping.from_yaml(
        CHARUCO_BOARD_MAPPING, known_tracker_keypoints=set(board.all_point_names)
    )
    assert mapping.directly_measured_landmark_names == frozenset(board.all_point_names)
    assert len(board.all_point_names) != len(
        CharucoBoardDefinition.create_letter_size_5x3().all_point_names
    )
