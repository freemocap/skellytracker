import numpy as np
import pytest
from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.tracker.detection_stage import DetectionStage
from skellytracker.core.tracker.tracker_state import StageState


def test_single_detector_result_is_not_concatenated(monkeypatch):
    points = Keypoints(names=('wrist',), xyz=np.ones((1, 3)), visibility=np.ones(1))
    def reject(*args):
        pytest.fail('A single detector result must not be concatenated')
    monkeypatch.setattr(Keypoints, 'concatenate', reject)
    merged, _ = DetectionStage(name='hand', keypoint_detectors=[])._smooth_keypoints([points], StageState(), 1 / 30)
    assert merged is points


def test_multiple_detector_results_are_joined_in_order():
    first = Keypoints(names=('wrist',), xyz=np.ones((1, 3)), visibility=np.ones(1))
    second = Keypoints(names=('index',), xyz=np.full((1, 3), 2.), visibility=np.ones(1))
    merged, _ = DetectionStage(name='hand', keypoint_detectors=[])._smooth_keypoints([first, second], StageState(), 1 / 30)
    assert merged.names == ('wrist', 'index')
    np.testing.assert_array_equal(merged.xyz, [[1., 1., 1.], [2., 2., 2.]])
