"""Serialized standalone detector recipes retain their concrete configuration."""
import subprocess
import sys


def test_fiducial_recipes_restore_without_consumer_imports():
    subprocess.run([sys.executable, '-B', '-c', """
from skellytracker.core.config.detection_stage_config import DetectionStageConfig
stage = DetectionStageConfig.model_validate({'name': 'board', 'keypoint_detectors': [
    {'detector_type': 'charuco', 'session_backend': 'cpu'},
    {'detector_type': 'aruco', 'session_backend': 'cpu', 'aruco_ids': [2, 5, 9]}]})
assert stage.keypoint_detectors[0].board.squares_x == 5
assert stage.keypoint_detectors[1].aruco_ids == (2, 5, 9)
assert DetectionStageConfig.model_validate_json(stage.model_dump_json(round_trip=True)) == stage
import sys
assert not any(name.split('.')[0] in {'freemocap', 'skellycam', 'skellyforge'} for name in sys.modules)
"""], check=True, capture_output=True, text=True, timeout=30)
