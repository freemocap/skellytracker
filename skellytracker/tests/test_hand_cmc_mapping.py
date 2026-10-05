"""CMC estimates use measured palm frames and retain constructed provenance.

Reference coordinates below are the standard-human hand's authored rest geometry
in reference-unit coordinates, expressed in an indexward/normal/longitudinal basis.
They are intentional fixtures, not a runtime dependency on a skeleton package.
"""

import numpy as np
import pytest

from skellytracker.core.io.mapping_paths import MEDIAPIPE_HAND_MAPPING, RTMPOSE_HAND_MAPPING
from skellytracker.core.io.tracker_mapping import TrackerMapping


CASES = [(RTMPOSE_HAND_MAPPING, 'middle_finger1', 'forefinger1'),
         (MEDIAPIPE_HAND_MAPPING, 'middle_finger_mcp', 'index_finger_mcp')]
FINGERS = ('index', 'middle', 'ring', 'pinky')


def palm(side, middle, index):
    mirror = 1 if side == 'left' else -1
    return {
        f'{side}_wrist': np.zeros(3),
        f'{side}_hand_{middle}': np.array([0., 0., .056489]),
        f'{side}_hand_{index}': np.array([mirror * .006682, 0., .054667]),
    }


@pytest.mark.parametrize('path,middle,index', CASES)
@pytest.mark.parametrize('side', ['left', 'right'])
def test_cmc_reference_geometry_and_similarity_equivariance(path, middle, index, side):
    mapping = TrackerMapping.from_yaml(path)
    points = palm(side, middle, index)
    before = {n: p.copy() for n, p in points.items()}
    result = mapping.apply(tracker_positions=points)
    mirror = 1 if side == 'left' else -1
    for finger, x in zip(FINGERS, [.006682, 0., -.006074, -.011541]):
        np.testing.assert_allclose(result[f'{side}_{finger}_cmc'], [mirror * x, 0., .015185], atol=1e-12)
    rotation = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    moved = mapping.apply(tracker_positions={n: 1700 * rotation @ p + [120., -90., 450.] for n, p in points.items()})
    for finger in FINGERS:
        name = f'{side}_{finger}_cmc'
        np.testing.assert_allclose(moved[name], 1700 * rotation @ result[name] + [120., -90., 450.], atol=1e-10)
        assert name not in mapping.directly_measured_landmark_names
    for name in points:
        np.testing.assert_array_equal(points[name], before[name])
    # Existing measured knuckle mapping remains direct and unchanged.
    np.testing.assert_array_equal(result[f'{side}_index_mcp'], points[f'{side}_hand_{index}'])


@pytest.mark.parametrize('path,middle,index', CASES)
@pytest.mark.parametrize('failure', ['wrist', 'middle', 'index', 'parallel', 'zero'])
def test_missing_or_degenerate_palm_omits_only_affected_side(path, middle, index, failure):
    mapping = TrackerMapping.from_yaml(path)
    points = palm('left', middle, index) | palm('right', middle, index)
    if failure in ('wrist', 'middle', 'index'):
        name = {'wrist': 'left_wrist', 'middle': f'left_hand_{middle}', 'index': f'left_hand_{index}'}[failure]
        del points[name]
    elif failure == 'parallel':
        points[f'left_hand_{index}'] = points[f'left_hand_{middle}'] * .8
    else:
        points[f'left_hand_{middle}'] = points['left_wrist'].copy()
    result = mapping.apply(tracker_positions=points)
    assert all(f'left_{finger}_cmc' not in result for finger in FINGERS)
    assert all(f'right_{finger}_cmc' in result for finger in FINGERS)


def test_detectors_agree_when_the_palm_deforms_and_quality_uses_all_inputs():
    outputs = []
    for path, middle, index in CASES:
        mapping = TrackerMapping.from_yaml(path)
        points = palm('left', middle, index)
        points[f'left_hand_{index}'] += [.002, .01, -.008]
        quality = {name: .9 for name in points}
        quality[f'left_hand_{index}'] = .3
        result = mapping.apply_with_quality(tracker_positions=points, tracker_quality=quality)
        for finger in FINGERS:
            assert result.quality[f'left_{finger}_cmc'] == .3
        outputs.append(result.positions)
    for finger in FINGERS:
        np.testing.assert_allclose(outputs[0][f'left_{finger}_cmc'], outputs[1][f'left_{finger}_cmc'], atol=1e-12)
