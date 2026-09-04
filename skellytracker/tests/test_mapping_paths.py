"""Tests for the light mapping-path registry module.

These tests ensure that the tracker→standard-human mapping YAML paths live in a
single import-light module and that the detector classmethods simply delegate to
it (one source of truth), so base-only consumers (e.g. skellyforge) can reach
the YAMLs without importing detector machinery.
"""

from __future__ import annotations

import subprocess
import sys

from pathlib import Path

import pytest

from skellytracker.core.io import mapping_paths


def test_module_imports_without_detector_machinery() -> None:
    """Importing the mapping_paths module must not pull in mediapipe/onnxruntime.

    The check runs in a fresh subprocess so that mediapipe/onnxruntime already
    imported by *other* tests in this session cannot pollute ``sys.modules``.
    """
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import skellytracker.core.io.mapping_paths; "
            "print('mediapipe' in sys.modules, 'onnxruntime' in sys.modules)",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False False"


@pytest.mark.parametrize(
    ("detector_module", "detector_class", "constant_name"),
    [
        (
            "skellytracker.core.detectors.keypoint_detectors.rtmpose.body.rtmpose_body_detector",
            "RTMPoseBodyDetector",
            "RTMPOSE_BODY_MAPPING",
        ),
        (
            "skellytracker.core.detectors.keypoint_detectors.rtmpose.hand.rtmpose_hand_detector",
            "RTMPoseHandDetector",
            "RTMPOSE_HAND_MAPPING",
        ),
        (
            "skellytracker.core.detectors.keypoint_detectors.mediapipe.body.mediapipe_pose_detector",
            "MediapipePoseKeypointDetector",
            "MEDIAPIPE_BODY_MAPPING",
        ),
        (
            "skellytracker.core.detectors.keypoint_detectors.mediapipe.hands.mediapipe_hand_detector",
            "MediapipeHandKeypointDetector",
            "MEDIAPIPE_HAND_MAPPING",
        ),
    ],
)
def test_detector_classmethod_matches_registry(
    detector_module: str, detector_class: str, constant_name: str
) -> None:
    module = pytest.importorskip(detector_module)
    detector = getattr(module, detector_class)
    assert detector.standard_human_mapping_path() == getattr(mapping_paths, constant_name)
