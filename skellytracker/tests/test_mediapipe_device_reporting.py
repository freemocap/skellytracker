"""The reported MediaPipe device must match the options given to native tasks."""
import importlib
import logging
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from skellytracker.core.sessions.mediapipe_session import MediaPipeSession, MediaPipeSessionConfig


@pytest.mark.parametrize('disable_gpu', [None, True])
def test_cpu_session_reports_cpu_without_model_probe(disable_gpu, caplog, monkeypatch):
    from skellytracker.core.detectors.keypoint_detectors.mediapipe import mediapipe_model_manager
    monkeypatch.setattr(mediapipe_model_manager, 'get_pose_model_path',
                        Mock(side_effect=AssertionError('Session creation must not load a probe model')))
    monkeypatch.setenv('MEDIAPIPE_DISABLE_GPU', 'external-setting')
    with caplog.at_level(logging.INFO):
        session = MediaPipeSession.create(MediaPipeSessionConfig(disable_gpu=disable_gpu))
    assert 'MediaPipe tasks use the CPU delegate.' in caplog.text
    assert 'GPU probe succeeded' not in caplog.text
    assert os.environ['MEDIAPIPE_DISABLE_GPU'] == 'external-setting'
    session.close()


@pytest.mark.parametrize('dataset', ['freemocap_test_data', 'freemocap_sample_data'])
def test_explicit_cpu_matches_previous_default_on_reference_video(dataset):
    """Three frames per camera: native CPU selection must not change predictions."""
    from dataclasses import replace
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from skellytracker.core.detectors.keypoint_detectors.mediapipe import mediapipe_model_manager as assets
    from skellytracker.core.detectors.keypoint_detectors.mediapipe.hands.mediapipe_hand_detector import (
        MediapipeHandDetectorConfig, MediapipeHandKeypointDetector,
    )

    root = Path(os.environ.get('SKELLYTRACKER_REFERENCE_ROOT',
                               str(Path.home() / 'freemocap_data/testing/prepared')))
    paths = sorted((root / dataset / 'current/recordings' / dataset / 'synchronized_videos').glob('*.mp4'))
    cached_model = assets.CACHE_DIR / assets._filename_from_url(assets.HAND_MODEL_URL)
    if not paths or not cached_model.exists():
        pytest.skip('Requires prepared reference recording and already cached hand model; never downloads')
    session = MediaPipeSession.create()
    for path in paths:
        with patch.object(vision.HandLandmarker, 'create_from_options',
                          wraps=vision.HandLandmarker.create_from_options) as create_task:
            current = MediapipeHandKeypointDetector.create(MediapipeHandDetectorConfig(), session)
        options = current.landmarker.options
        previous_options = replace(options, base_options=BaseOptions(model_asset_path=str(cached_model)))
        capture = cv2.VideoCapture(str(path))
        try:
            with vision.HandLandmarker.create_from_options(previous_options) as previous:
                frame_rate = capture.get(cv2.CAP_PROP_FPS)
                assert frame_rate > 0
                for frame_number in range(3):
                    success, frame = capture.read()
                    assert success
                    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    timestamp_ms = round(frame_number * 1000 / frame_rate)
                    assert current.landmarker.detect_for_video(image, timestamp_ms) == previous.detect_for_video(image, timestamp_ms)
        finally:
            current.close()
            capture.release()
    session.close()


def test_explicit_unsupported_gpu_request_fails():
    with pytest.raises(ValueError, match='GPU execution is not implemented'):
        MediaPipeSession.create(MediaPipeSessionConfig(disable_gpu=False))


@pytest.mark.parametrize('folder, name, task', [
    ('hands', 'Hand', 'HandLandmarker'),
    ('body', 'Pose', 'PoseLandmarker'),
    ('face', 'Face', 'FaceLandmarker'),
])
@pytest.mark.parametrize('mode', ['image', 'video'])
def test_each_native_task_explicitly_uses_cpu(folder, name, task, mode, monkeypatch):
    from mediapipe.tasks.python import BaseOptions, vision
    module = importlib.import_module(
        f'skellytracker.core.detectors.keypoint_detectors.mediapipe.{folder}.mediapipe_{name.lower()}_detector')
    monkeypatch.setattr(module, f'get_{name.lower()}_model_path', lambda *args: Path('unused.task'))
    create_task = Mock()
    monkeypatch.setattr(getattr(vision, task), 'create_from_options', create_task)
    session = MediaPipeSession.create(MediaPipeSessionConfig(running_mode=mode))
    detector = getattr(module, f'Mediapipe{name}KeypointDetector').create(
        getattr(module, f'Mediapipe{name}DetectorConfig')(), session)
    create_task.assert_not_called()
    options = detector.landmarker.options
    assert options.base_options.delegate == BaseOptions.Delegate.CPU
    assert options.running_mode == (vision.RunningMode.IMAGE if mode == 'image' else vision.RunningMode.VIDEO)
    detector.landmarker.detect(object())
    detector.close()
    detector.close()
    create_task.return_value.close.assert_called_once()
    session.close()
