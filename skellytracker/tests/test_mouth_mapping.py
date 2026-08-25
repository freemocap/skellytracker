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
import pytest

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


def _tpose_tracker_head() -> dict[str, np.ndarray]:
    """Tracker keypoints standing in at the authored T-pose.

    Derived from SkellyForge's rest pose itself rather than hand-copied
    coordinates - hand-copied numbers drift silently when the model moves on.
    The head-frame offsets consume the ears/eyes/nose plus the shoulders and
    hips that anchor the trunk frame, mirroring the table in skellyforge's
    ``test_tracker_mapping_offset_round_trip.py``.
    """
    pytest.importorskip("skellyforge", reason="skellyforge is a dev dependency")
    from skellyforge.core.skeleton.pose.rest_pose import RestPose
    from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition

    skeleton = SkeletonDefinition.from_default_yaml()
    landmark_positions = RestPose.from_default_yaml(skeleton=skeleton).landmark_positions
    standins = {
        "nose": "nose",
        "left_eye": "left_eye",
        "right_eye": "right_eye",
        "left_ear": "left_ear",
        "right_ear": "right_ear",
        "left_shoulder": "left_acromion",
        "right_shoulder": "right_acromion",
        "left_hip": "left_hip_socket",
        "right_hip": "right_hip_socket",
    }
    return {
        tracker_name: landmark_positions[landmark_name].array
        for tracker_name, landmark_name in standins.items()
    }


def test_both_body_mappings_load_with_known_tracker_keypoints():
    for mapping_path in (MEDIAPIPE_BODY_MAPPING, RTMPOSE_BODY_MAPPING):
        mapping = TrackerMapping.from_yaml(Path(mapping_path))
        TrackerMapping.from_yaml(
            Path(mapping_path),
            known_tracker_keypoints=_known_keypoints(mapping, mapping_path),
        )


def test_head_landmarks_land_on_rest_positions():
    pytest.importorskip("skellyforge", reason="skellyforge is a dev dependency")
    from skellyforge.core.skeleton.pose.rest_pose import RestPose
    from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition

    skeleton = SkeletonDefinition.from_default_yaml()
    landmark_positions = RestPose.from_default_yaml(skeleton=skeleton).landmark_positions
    tracker_body = _tpose_tracker_head()

    _tolerance_mm = 2.0
    for mapping_path in (MEDIAPIPE_BODY_MAPPING, RTMPOSE_BODY_MAPPING):
        mapping = TrackerMapping.from_yaml(Path(mapping_path))
        result = mapping.apply(tracker_positions=tracker_body)

        # head_center is the ear midpoint (the tracker's head-center approximation).
        assert np.allclose(
            result["head_center"],
            (
                np.asarray(landmark_positions["left_ear"].array)
                + np.asarray(landmark_positions["right_ear"].array)
            )
            / 2.0,
            atol=0.1,
        )
        # Everything else is derived in the head frame, so it must land on the
        # authored skull rest positions.
        for landmark_name in ("head_vertex", "chin", "left_canine_tooth_tip", "right_canine_tooth_tip"):
            error_mm = float(
                np.linalg.norm(
                    np.asarray(result[landmark_name])
                    - np.asarray(landmark_positions[landmark_name].array)
                )
            )
            assert error_mm <= _tolerance_mm, (
                f"{mapping_path.name}: {landmark_name}: {error_mm:.2f} mm from its "
                f"authored rest position (allowed {_tolerance_mm} mm) - "
                "regenerate the mapping ratios"
            )
