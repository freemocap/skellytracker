"""Mouth-corner keypoint mapping tests.

The standard human's segment model gains two mouth corners — ``left_mouth`` and
``right_mouth``.  MediaPipe pose actually tracks them (``mouth_left`` /
``mouth_right``), so those map 1:1 for real; RTMPose (COCO-17) tracks no mouth
point, so its corners are derived via ``anatomical_offset`` from the nose, in the
same frame as the existing ``jaw`` entry.
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

# Left–right handedness of the synthetic pose follows the canonical standard-human
# frame: +X forward (anterior), +Y LEFT, +Z up.  (right-handed: X × Y = Z)


_SCHEMA_NAME = {
    MEDIAPIPE_BODY_MAPPING: "mediapipe_body.yaml",
    RTMPOSE_BODY_MAPPING: "rtmpose_body.yaml",
}


def _known_keypoints(mapping: TrackerMapping, mapping_path: str) -> set[str]:
    """The full set of names a self-consistent load references.

    D24 checks that every tracker-side name the mapping references is a name the
    tracker produces.  A mapping references its own intermediate standard-human
    names (``head_center``, ``neck_center``, ``hips_center``) from its
    ``anatomical_offset`` frames, so the "known" set is the detector's raw tracked
    points (from the schema YAML) *plus* every name the mapping itself produces.
    """
    raw = set(load_point_names(Path(mapping_path).with_name(_SCHEMA_NAME[mapping_path])))
    return raw | set(mapping.keypoint_names)


def test_both_body_mappings_load_with_known_tracker_keypoints():
    # D24: every tracker-side name each mapping references must be a name the
    # tracker produces.  The new mouth-corner names (``mouth_left`` /
    # ``mouth_right`` in mediapipe; ``left_mouth`` / ``right_mouth`` derived in
    # rtmpose, referencing nothing new) are known — loading must not raise.
    for mapping_path in (MEDIAPIPE_BODY_MAPPING, RTMPOSE_BODY_MAPPING):
        mapping = TrackerMapping.from_yaml(Path(mapping_path))
        TrackerMapping.from_yaml(
            Path(mapping_path),
            known_tracker_keypoints=_known_keypoints(mapping, mapping_path),
        )


def test_mediapipe_body_maps_mouth_corners_for_real():
    mapping = TrackerMapping.from_yaml(Path(MEDIAPIPE_BODY_MAPPING))
    frame = {
        "mouth_left": np.array([1.0, 2.0, 3.0]),
        "mouth_right": np.array([4.0, 5.0, 6.0]),
    }
    result = mapping.apply(frame)
    assert np.allclose(result["left_mouth"], [1.0, 2.0, 3.0])
    assert np.allclose(result["right_mouth"], [4.0, 5.0, 6.0])


def _rtmpose_upright_frame() -> dict[str, np.ndarray]:
    # An upright, symmetric pose with the subject facing +X (anterior), +Z up,
    # +Y = left.  Eyes symmetric about the midline in Y, eye_width = 2.0,
    # head_center directly above neck_center so `up` is pure +Z.
    return {
        "nose": np.array([0.0, 0.0, 0.0]),
        "left_eye": np.array([0.0, 1.0, 0.0]),
        "right_eye": np.array([0.0, -1.0, 0.0]),
        "left_ear": np.array([0.0, 1.0, 0.5]),
        "right_ear": np.array([0.0, -1.0, 0.5]),
        "left_shoulder": np.array([0.0, 1.0, -1.0]),
        "right_shoulder": np.array([0.0, -1.0, -1.0]),
    }


def test_rtmpose_body_derives_mouth_corners():
    mapping = TrackerMapping.from_yaml(Path(RTMPOSE_BODY_MAPPING))
    frame = _rtmpose_upright_frame()
    result = mapping.apply(frame)

    nose = frame["nose"]
    eye_width = np.linalg.norm(frame["left_eye"] - frame["right_eye"])
    assert np.isclose(eye_width, 2.0)

    left_mouth = result["left_mouth"]
    right_mouth = result["right_mouth"]

    # Both below the nose (lower Z).
    assert left_mouth[2] < nose[2]
    assert right_mouth[2] < nose[2]

    # Both posterior to the nose (toward -X, since the pose faces +X).
    assert left_mouth[0] < nose[0]
    assert right_mouth[0] < nose[0]

    # left_mouth toward the left eye (+Y), right_mouth toward the right eye (-Y);
    # their Y positions straddle the nose's Y.
    assert left_mouth[1] > nose[1]
    assert right_mouth[1] < nose[1]

    # Magnitude of (mouth - nose) ~= eye_width * sqrt(0.35² + 0.2² + 0.3²).
    expected_mag = eye_width * np.sqrt(0.35**2 + 0.2**2 + 0.3**2)
    assert np.isclose(np.linalg.norm(left_mouth - nose), expected_mag, atol=1e-6)
    assert np.isclose(np.linalg.norm(right_mouth - nose), expected_mag, atol=1e-6)
