"""Which mapped landmarks are measurements, and which are constructed from the template.

Downstream, SkellyForge's body-scale fit infers the subject's SIZE from mapped landmarks,
and it must not take an ``anatomical_offset`` landmark as evidence: those sit at
``ratio x reference_length`` along an authored direction, so the distance between two of
them is an authored number times a span measured elsewhere. This test pins the split so a
new mapping entry cannot quietly join the wrong side of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from skellytracker.core.io.mapping_paths import (
    MEDIAPIPE_BODY_MAPPING,
    RTMPOSE_BODY_MAPPING,
)
from skellytracker.core.io.tracker_mapping import TrackerMapping

_BODY_MAPPINGS = (MEDIAPIPE_BODY_MAPPING, RTMPOSE_BODY_MAPPING)


@pytest.mark.parametrize("mapping_path", _BODY_MAPPINGS, ids=lambda p: p.name)
def test_offset_landmarks_are_not_reported_as_measured(mapping_path) -> None:
    mapping = TrackerMapping.from_yaml(mapping_path)
    measured = mapping.directly_measured_landmark_names

    assert measured, "a body mapping must measure something"
    for constructed in ("chest_center", "xiphoid_process", "neck_center"):
        assert constructed not in measured, (
            f"{constructed!r} is an anatomical_offset - it is placed by the template, not "
            "measured, so it must not be offered as evidence of the subject's size"
        )
    # The keypoints the tracker actually produces are.
    for real in ("left_knee", "left_ankle", "left_elbow", "left_wrist"):
        assert real in measured


@pytest.mark.parametrize("mapping_path", _BODY_MAPPINGS, ids=lambda p: p.name)
def test_measured_landmarks_carry_the_subjects_scale(mapping_path) -> None:
    """The property that makes the split meaningful, checked rather than asserted in prose.

    Every non-offset form is an affine combination of measured keypoints with constant
    coefficients, so scaling the subject scales those landmarks by exactly the same factor.
    Drive the same mapping with a subject and with a subject twice the size: the measured
    landmarks double, and a distance between two of them is therefore a real distance ON
    the subject.
    """
    mapping = TrackerMapping.from_yaml(mapping_path)
    rng = np.random.default_rng(7)
    keypoint_names = sorted(mapping._referenced_tracker_names())
    small = {name: rng.normal(size=3) * 500.0 for name in keypoint_names}
    large = {name: 2.0 * position for name, position in small.items()}

    small_output = mapping.apply(tracker_positions=small)
    large_output = mapping.apply(tracker_positions=large)

    for name in mapping.directly_measured_landmark_names:
        if name not in small_output:
            continue
        np.testing.assert_allclose(
            large_output[name], 2.0 * small_output[name], atol=1e-9, err_msg=name
        )
