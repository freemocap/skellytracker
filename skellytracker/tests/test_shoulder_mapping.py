"""Shoulder attachment geometry is symmetric and detector-independent."""

import numpy as np
import pytest

from skellytracker.core.io.mapping_paths import MEDIAPIPE_BODY_MAPPING, RTMPOSE_BODY_MAPPING
from skellytracker.core.io.tracker_mapping import TrackerMapping


ATTACHMENTS = ('left_sternoclavicular', 'right_sternoclavicular', 'sternoclavicular_notch')


def shoulders():
    return {
        'left_shoulder': np.array([-200., 0., 600.]),
        'right_shoulder': np.array([200., 0., 600.]),
        'left_hip': np.array([-150., 0., 0.]),
        'right_hip': np.array([150., 0., 0.]),
    }


@pytest.mark.parametrize('path', [MEDIAPIPE_BODY_MAPPING, RTMPOSE_BODY_MAPPING])
def test_attachments_are_symmetric_and_transform_with_the_subject(path):
    mapping = TrackerMapping.from_yaml(path)
    points = shoulders()
    mapped = mapping.apply(tracker_positions=points)
    left, right, center = (mapped[n] for n in ATTACHMENTS)
    np.testing.assert_allclose(left * [-1., 1., 1.], right, atol=1e-10)
    np.testing.assert_allclose((left + right) / 2, center, atol=1e-10)
    # Rotation, translation and size changes must not alter attachment proportions.
    rotation = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    translation = np.array([123., -57., 800.])
    moved = mapping.apply(tracker_positions={n: 1.7 * rotation @ p + translation for n, p in points.items()})
    for name in ATTACHMENTS:
        np.testing.assert_allclose(moved[name], 1.7 * rotation @ mapped[name] + translation, atol=1e-10)


@pytest.mark.parametrize('raise_left', [0., 150.])
def test_trackers_agree_for_identical_shoulder_and_hip_evidence(raise_left):
    points = shoulders()
    points['left_shoulder'][2] += raise_left
    results = [TrackerMapping.from_yaml(path).apply(tracker_positions=points)
               for path in (MEDIAPIPE_BODY_MAPPING, RTMPOSE_BODY_MAPPING)]
    for name in (*ATTACHMENTS, 'neck_center', 'chest_center', 'xiphoid_process'):
        np.testing.assert_allclose(results[0][name], results[1][name], atol=1e-10)


@pytest.mark.parametrize('path', [MEDIAPIPE_BODY_MAPPING, RTMPOSE_BODY_MAPPING])
def test_missing_shoulder_does_not_fabricate_attachments(path):
    points = shoulders()
    del points['left_shoulder']
    result = TrackerMapping.from_yaml(path).apply(tracker_positions=points)
    assert all(name not in result for name in ATTACHMENTS)
