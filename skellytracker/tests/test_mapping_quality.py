"""Mapping quality follows all required measurement dependencies."""

import numpy as np
import pytest

from skellytracker.core.io.composite_tracker_mapping import CompositeTrackerMapping
from skellytracker.core.io.tracker_mapping import TrackerMapping


def test_convex_mapping_does_not_average_away_poor_evidence() -> None:
    mapping = TrackerMapping(entries={"direct": "a", "center": ["a", "b"], "weighted": {"a": 0.9, "b": 0.1}})
    positions = {"a": np.zeros(3), "b": np.ones(3)}
    result = mapping.apply_with_quality(tracker_positions=positions, tracker_quality={"a": 0.9, "b": 0.2})
    assert result.quality == {"direct": 0.9, "center": 0.2, "weighted": 0.2}
    for name, expected in mapping.apply(tracker_positions=positions).items():
        np.testing.assert_array_equal(result.positions[name], expected)
    missing = mapping.apply_with_quality(tracker_positions={"a": positions["a"]}, tracker_quality={"a": 0.9})
    assert set(missing.positions) == {"direct"}


def test_passthrough_prefix_and_composition() -> None:
    mapping = CompositeTrackerMapping(mappings=(
        TrackerMapping(entries={}, prefix="board_", passthrough_keypoints_as_landmarks=True,
                       known_tracker_keypoints={"board_corner"}),
        TrackerMapping(entries={"head": "nose"}),
    ))
    result = mapping.apply_with_quality(
        tracker_positions={"board_corner": np.ones(3), "nose": np.zeros(3)},
        tracker_quality={"board_corner": 0.8, "nose": 0.6},
    )
    assert result.quality == {"corner": 0.8, "head": 0.6}


def test_anatomical_offset_quality_includes_frame_and_mapped_origin() -> None:
    mapping = TrackerMapping(entries={
        "center": ["left", "right"],
        "offset": {
            "form": "anatomical_offset", "origin": ["center"],
            "frame": {
                "lateral": {"from": "left", "to": "right", "kind": "exact"},
                "up": {"from": "center", "to": "top", "kind": "approximate"},
            },
            "offset": {"up": 0.1}, "reference_length": {"from": "left", "to": "right"},
        },
    })
    result = mapping.apply_with_quality(
        tracker_positions={"left": np.array([-1., 0., 0.]), "right": np.array([1., 0., 0.]), "top": np.array([0., 0., 1.])},
        tracker_quality={"left": 0.9, "right": 0.8, "top": 0.3},
    )
    assert result.quality == {"center": 0.8, "offset": 0.3}


@pytest.mark.parametrize("score", [float("nan"), -0.1, 1.1])
def test_invalid_quality_fails(score: float) -> None:
    with pytest.raises(ValueError, match="quality"):
        TrackerMapping(entries={"head": "nose"}).apply_with_quality(
            tracker_positions={"nose": np.zeros(3)}, tracker_quality={"nose": score},
        )


def test_missing_quality_is_a_contract_error() -> None:
    with pytest.raises(ValueError, match="matching quality"):
        TrackerMapping(entries={"head": "nose"}).apply_with_quality(
            tracker_positions={"nose": np.zeros(3)}, tracker_quality={},
        )
