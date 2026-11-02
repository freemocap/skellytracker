"""Native CPU factory checks for the opt-in dependent hand detector; no downloads."""
import os
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest

from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.detectors.keypoint_detectors.mediapipe_onnx.hand_detector import OnnxHandDetectorConfig
from skellytracker.core.detectors.keypoint_detectors.mediapipe_onnx.model_preparation import CANDIDATE_HASHES
from skellytracker.core.detectors.keypoint_detectors.mediapipe_onnx.candidate_models import candidate_sidecar
from skellytracker.core.tracker.tracker_factory import build_tracker, collect_required_sessions
from skellytracker.core.tracker.tracker_state import TrackerState

@pytest.fixture
def cpu_tracker():
    root=Path(os.environ.get('SKELLYTRACKER_HAND_CANDIDATE_DIR','.test-artifacts/w4/upstream')).resolve()
    if not all((root/name).is_file() for name in CANDIDATE_HASHES):
        pytest.skip('Pinned candidate weights not supplied')
    # Zero gates exercise both native networks on synthetic images. This checks
    # execution/coordinates, not detection quality or production thresholds.
    config=TrackerConfig(stages=[DetectionStageConfig(name='hand',keypoint_detectors=[OnnxHandDetectorConfig(
        assumed_handedness='left',model_directory=root,min_hand_detection_confidence=0.,min_hand_presence_confidence=0.)])])
    assert len(collect_required_sessions(config).onnx_model_specs)==2
    tracker=build_tracker(config,batch_size=3,onnx_overrides={'execution_provider':'cpu','fp16':False,'intra_op_num_threads':1})
    try:
        yield tracker
    finally:
        tracker.close()
        assert not tracker.sessions['onnx']._sessions


def test_native_factory_single_batch_partial_batches_and_owned_results(cpu_tracker):
    session=cpu_tracker.sessions['onnx']
    assert session.execution_provider=='cpu'
    assert all(native.get_providers()==['CPUExecutionProvider'] for native in session._sessions.values())
    generator=np.random.default_rng(52)
    retained=[]
    for count in (1,3,2,1):
        images={f'camera_{i}':generator.integers(0,256,(125+12*i,140+11*i,3),dtype=np.uint8) for i in range(count)}
        batch,states=cpu_tracker.process_batch(images,frame_number=count,states={},timestamp_ms=1000+count)
        for camera,image in images.items():
            single,_=cpu_tracker.process_image(image,frame_number=count,state=TrackerState(),timestamp_ms=1000+count)
            actual=batch[camera].stages['hand'].keypoints
            expected=single.stages['hand'].keypoints
            assert actual.names==expected.names
            np.testing.assert_allclose(actual.xyz,expected.xyz,rtol=1e-4,atol=1e-3,equal_nan=True)
            np.testing.assert_array_equal(actual.visibility,expected.visibility)
            assert states[camera].stage_states['hand'].last_timestamp_ms==1000+count
            retained.append((actual,actual.xyz.copy(),actual.visibility.copy()))
    assert any(points.n_valid for points,_,_ in retained), 'Synthetic native run must exercise the hand output branch'
    cpu_tracker.reset_temporal_state()
    for points,xyz,visibility in retained:
        np.testing.assert_array_equal(points.xyz,xyz)
        np.testing.assert_array_equal(points.visibility,visibility)


def test_native_detector_skips_zero_valid_crops(cpu_tracker,monkeypatch):
    detector=cpu_tracker.stages[0].keypoint_detectors[0]
    run=Mock(side_effect=AssertionError('No inference for empty crops'))
    monkeypatch.setattr(detector.session,'run_batched',run)
    assert detector.detect_batch({})=={}
    points=detector.detect_batch({'empty_camera':np.empty((0,0,3),dtype=np.uint8)})['empty_camera']
    assert np.isnan(points.xyz).all() and not points.visibility.any()
    run.assert_not_called()


def test_prepared_sidecars_describe_actual_input_and_named_outputs():
    hand=candidate_sidecar('hand')
    palm=candidate_sidecar('palm')
    assert hand.pose.requires_object_detector and hand.input.resize.method=='stretch'
    assert hand.pose.decode.coordinate_scale=='pixel'
    assert hand.outputs[0].fields==['x','y','z','hand_presence','handedness','world_x','world_y','world_z']
    assert palm.decode.score_field=='score_logit'
    assert hand.pose.tracked_points[0]=='wrist'


def test_coordinate_sidecar_does_not_enable_generic_simcc_runtime():
    from skellytracker.core.detectors.processing.simcc_decode import simcc_pose_decode
    with pytest.raises(NotImplementedError, match="coordinate"):
        simcc_pose_decode(
            raw=[], center=np.zeros(2), scale=np.ones(2), model_input_size=(224, 224),
            decode_spec=candidate_sidecar("hand").pose.decode,
        )
