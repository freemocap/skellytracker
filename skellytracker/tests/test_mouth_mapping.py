"""Head/face keypoint mapping tests.

The standard human's skull names the canine tooth tips (left_canine_tooth_tip /
right_canine_tooth_tip) as its mouth-region landmarks, plus the chin and head
vertex. Neither MediaPipe pose (which tracks the lip commissures mouth_left /
mouth_right, not the tooth tips) nor RTMPose (no mouth point at all) tracks them
directly; both derive them via anatomical_offset from the nose in the head frame,
so the head frame is checked against the SkellyForge rest positions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from skellytracker.core.detectors.keypoint_detectors._schema_loader import (
    load_point_names,
)
from skellytracker.core.io.mapping_paths import (
    MEDIAPIPE_BODY_MAPPING,
    RTMPOSE_BODY_MAPPING,
)
from skellytracker.core.io.tracker_mapping import TrackerMapping

_SCHEMA_NAME = {
    MEDIAPIPE_BODY_MAPPING: "mediapipe_body.yaml",
    RTMPOSE_BODY_MAPPING: "rtmpose_body.yaml",
}


def _known_keypoints(mapping: TrackerMapping, mapping_path: str) -> set[str]:
    """The full set of names a self-consistent load references."""
    raw = set(load_point_names(Path(mapping_path).with_name(_SCHEMA_NAME[mapping_path])))
    return raw | set(mapping.landmark_names)


def _rest_pose_keypoints() -> dict[str, np.ndarray]:
    """The tracker keypoints at the SkellyForge rest (T) pose, Blender world axes."""
    return {
        "nose": np.array([0.0, 65.0, 716.0]),
        "left_eye": np.array([-32.0, 30.0, 756.0]),
        "right_eye": np.array([32.0, 30.0, 756.0]),
        "left_ear": np.array([-70.0, -45.0, 711.0]),
        "right_ear": np.array([70.0, -45.0, 711.0]),
        "left_shoulder": np.array([-160.0, 0.0, 470.0]),
        "right_shoulder": np.array([160.0, 0.0, 470.0]),
        "left_hip": np.array([-88.0, 0.0, 0.0]),
        "right_hip": np.array([88.0, 0.0, 0.0]),
    }


def test_both_body_mappings_load_with_known_tracker_keypoints():
    for mapping_path in (MEDIAPIPE_BODY_MAPPING, RTMPOSE_BODY_MAPPING):
        mapping = TrackerMapping.from_yaml(Path(mapping_path))
        TrackerMapping.from_yaml(
            Path(mapping_path),
            known_tracker_keypoints=_known_keypoints(mapping, mapping_path),
        )


def test_head_landmarks_land_on_rest_positions():
    for mapping_path in (MEDIAPIPE_BODY_MAPPING, RTMPOSE_BODY_MAPPING):
        mapping = TrackerMapping.from_yaml(Path(mapping_path))
        result = mapping.apply(_rest_pose_keypoints())

        # head_center is the ear midpoint (the tracker's head-center approximation,
        # ~11 mm posterior-inferior of the SkellyForge skull origin).
        assert np.allclose(result["head_center"], [0.0, -45.0, 711.0], atol=0.1)
        # head_vertex sits ~130 mm above head_center along the head vertical.
        assert np.allclose(result["head_vertex"] - result["head_center"], [0.0, 0.0, 130.0], atol=7.0)
        # chin and canine tips are derived from the (directly tracked) nose, so they
        # land near the SkellyForge skull rest positions.
        assert np.allclose(result["chin"], [0.0, 60.0, 621.0], atol=5.0)
        assert np.allclose(result["left_canine_tooth_tip"], [-20.0, 25.0, 666.0], atol=5.0)
        assert np.allclose(result["right_canine_tooth_tip"], [20.0, 25.0, 666.0], atol=5.0)
