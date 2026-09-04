"""Which mapped landmarks are measurements, and which are constructed from the template.

A non-offset form (passthrough, mean, weighted sum) is an affine combination of measured
keypoints, so it carries the subject's real geometry; an ``anatomical_offset`` restates
the template. ``directly_measured_landmark_names`` draws that line, and downstream only
measured landmarks may set a fitted scale. These tests exercise that classification with
synthetic mappings, not the shipped files.
"""

from __future__ import annotations

import numpy as np

from skellytracker.core.io.tracker_mapping import TrackerMapping

_KEYPOINTS = {"left_knee", "left_hip", "right_hip", "left_ear", "right_ear", "left_shoulder"}


def _synthetic_mapping() -> TrackerMapping:
    return TrackerMapping(
        entries={
            "left_knee": "left_knee",  # passthrough -> measured
            "pelvis_origin": ["left_hip", "right_hip"],  # mean -> measured
            "head_center": {"left_ear": 0.5, "right_ear": 0.5},  # weighted -> measured
            "hip_center": {  # anatomical_offset -> constructed
                "form": "anatomical_offset",
                "origin": ["left_hip", "right_hip"],
                "frame": {
                    "up": {"from": "left_hip", "to": "left_shoulder", "kind": "exact"},
                    "lateral": {"from": "left_hip", "to": "right_hip", "kind": "approximate"},
                },
                "offset": {"up": 0.5},
                "reference_length": {"from": "left_hip", "to": "left_shoulder"},
            },
        },
        known_tracker_keypoints=_KEYPOINTS,
    )


def test_offsets_are_constructed_not_measured() -> None:
    mapping = _synthetic_mapping()
    assert mapping.directly_measured_landmark_names == frozenset(
        {"left_knee", "pelvis_origin", "head_center"}
    )
    assert "hip_center" not in mapping.directly_measured_landmark_names


def test_measured_landmarks_carry_the_subjects_scale() -> None:
    """Measured forms are affine with constant coefficients, so scaling the subject
    scales those landmarks by exactly the same factor."""
    mapping = _synthetic_mapping()
    rng = np.random.default_rng(7)
    small = {name: rng.normal(size=3) * 500.0 for name in sorted(_KEYPOINTS)}
    large = {name: 2.0 * position for name, position in small.items()}

    small_output = mapping.apply(tracker_positions=small)
    large_output = mapping.apply(tracker_positions=large)

    for name in mapping.directly_measured_landmark_names:
        np.testing.assert_allclose(
            large_output[name], 2.0 * small_output[name], atol=1e-9, err_msg=name
        )
