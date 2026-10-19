"""Loss recovery through the real stage, including batched ONNX scheduling."""
from unittest.mock import Mock

import numpy as np
import pytest

from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import KeypointDetector, ObjectDetector
from skellytracker.core.sessions.session import Session
from skellytracker.core.temporal_processing.bbox_policy import BBoxPolicy
from skellytracker.core.temporal_processing.keypoint_filtering import OneEuroFilter
from skellytracker.core.temporal_processing.keypoint_reset_policy import KeypointResetPolicy
from skellytracker.core.tracker.detection_stage import DetectionStage
from skellytracker.core.tracker.tracker_state import StageState


def _points(image):
    if image.flat[0] == 0:
        return Keypoints(names=("a", "b"), xyz=np.array([[np.nan, np.nan, 0.], [np.nan, np.nan, 0.]]),
                         visibility=np.zeros(2))
    return Keypoints(names=("a", "b"), xyz=np.array([[10., 10., 0.], [30., 30., 0.]]),
                     visibility=np.ones(2))


@pytest.fixture(params=["single", "batch-onnx", "batch-cpu"])
def runner(request):
    session = Mock(spec=Session)
    session.kind = "onnx" if request.param == "batch-onnx" else "cpu"
    session.run_batched = Mock(side_effect=lambda model, tensors: tensors)
    obj = Mock(spec=ObjectDetector)
    obj.session = session
    obj.config = Mock(model_name="object")
    def boxes(image):
        return [] if image.flat[0] == 2 else [BoundingBox.full_image(100, 100)]
    obj.detect.side_effect = lambda image, *args: boxes(image)
    obj.preprocess.side_effect = lambda image: (image, None)
    obj.postprocess.side_effect = lambda raw, meta: boxes(raw)
    kp = Mock(spec=KeypointDetector)
    kp.session = session
    kp.config = Mock(model_name="pose")
    kp.detect.side_effect = lambda image, *args: _points(image)
    kp.preprocess.side_effect = lambda image: (image, None)
    kp.postprocess.side_effect = lambda raw, meta: _points(raw)
    stage = DetectionStage(name="body", object_detector=obj, keypoint_detectors=[kp],
                           bbox_policy=BBoxPolicy(redetect_interval=150, keypoint_bbox_expansion=0.05))
    stage._cam_kp_detectors[0] = {"a": kp, "b": kp}
    states = {}

    def run(frame, value, other=None):
        images = {"a": np.full((100, 100, 3), value, dtype=np.uint8)}
        if other is not None:
            images["b"] = np.full((100, 100, 3), other, dtype=np.uint8)
        context = DetectionContext(frame_number=frame)
        if request.param == "single":
            obs, state = stage.run(images["a"], states.get("a", StageState()), context=context)
            observations, updated = {"a": obs}, {"a": state}
        else:
            observations, updated = stage.run_batch(images, states, context)
        states.update(updated)
        return observations

    yield stage, run, states, request.param
    stage.close()


def test_lost_pose_redetects_next_frame_then_resumes_reuse(runner):
    stage, run, states, _ = runner
    # A filter may hold the old keypoints; a detector reset may clear miss counters.
    # Neither should hide the fact that the latest measurement was empty.
    stage.keypoint_filter = OneEuroFilter(min_cutoff=1., beta=0., d_cutoff=1., max_gap_frames=5)
    stage.keypoint_reset_policy = KeypointResetPolicy(max_consecutive_misses=1)
    assert run(0, 1)["a"].detector_ran
    assert not run(1, 1)["a"].detector_ran
    assert not run(2, 0)["a"].detector_ran
    assert states["a"].last_keypoints.n_valid > 0
    assert run(3, 1)["a"].detector_ran
    assert not run(4, 1)["a"].detector_ran
    assert run(153, 1)["a"].detector_ran


def test_failed_person_detection_retries_despite_pose_output(runner):
    _, run, _, _ = runner
    assert run(0, 2)["a"].detector_ran
    assert run(1, 2)["a"].detector_ran
    assert run(2, 1)["a"].detector_ran
    assert not run(3, 1)["a"].detector_ran


@pytest.mark.parametrize("runner", ["batch-onnx", "batch-cpu"], indirect=True)
def test_one_lost_camera_triggers_synchronized_onnx_redetection(runner):
    _, run, _, mode = runner
    run(0, 1, 1)
    run(1, 0, 1)
    obs = run(2, 1, 1)
    assert obs["a"].detector_ran
    assert obs["b"].detector_ran == (mode == "batch-onnx")


def test_missing_pose_keeps_searching_until_reacquired(runner):
    _, run, _, _ = runner
    run(0, 1)
    run(1, 0)
    assert run(2, 0)["a"].detector_ran
    assert run(3, 0)["a"].detector_ran
    assert run(4, 1)["a"].detector_ran
    assert not run(5, 1)["a"].detector_ran


def test_object_only_stage_reuses_success_but_retries_failed_detection(runner):
    stage, run, _, _ = runner
    stage.keypoint_detectors = []
    assert run(0, 1)["a"].detector_ran
    assert not run(1, 1)["a"].detector_ran
    assert run(150, 2)["a"].detector_ran
    assert run(151, 1)["a"].detector_ran
    assert not run(152, 1)["a"].detector_ran


@pytest.mark.parametrize("invalid", ["low-confidence", "collapsed", "nonfinite"])
def test_unusable_tracking_crop_requests_detection(runner, invalid):
    stage, run, _, _ = runner
    run(0, 1)
    kpts = _points(np.ones((100, 100, 3), dtype=np.uint8))
    stage.bbox_policy.keypoint_bbox_min_visibility = 0.4
    if invalid == "low-confidence":
        kpts.visibility[:] = 0.1
    elif invalid == "collapsed":
        kpts.xyz[:] = 10.
    else:
        kpts.xyz[:] = np.inf
    kp = stage.keypoint_detectors[0]
    kp.detect.side_effect = lambda *args: kpts
    kp.postprocess.side_effect = lambda *args: kpts
    assert not run(1, 1)["a"].detector_ran
    assert run(2, 1)["a"].detector_ran
