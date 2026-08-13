"""The tracker→standard-human mapping YAML paths.

This module exists so that light consumers (skellyforge, a base-only install
with no extras) can reach the tracker→standard-human mapping YAML files without
importing any detector machinery (mediapipe / onnxruntime / torch trees).

The four detector classes that produce a mapping path simply delegate to the
constants defined here — this is the single source of truth for those paths, so
the strings are never duplicated.

Keep this module import-light: no detector imports, no heavy dependencies. Only
``pathlib`` (and the minimal standard library) is used.
"""

from __future__ import annotations

from pathlib import Path

RTMPOSE_BODY_MAPPING = Path(__file__).parent.parent / "detectors" / "keypoint_detectors" / "rtmpose" / "body" / "rtmpose_body_to_standard_human_mapping.yaml"
RTMPOSE_HAND_MAPPING = Path(__file__).parent.parent / "detectors" / "keypoint_detectors" / "rtmpose" / "hand" / "rtmpose_hand_to_standard_human_mapping.yaml"
MEDIAPIPE_BODY_MAPPING = Path(__file__).parent.parent / "detectors" / "keypoint_detectors" / "mediapipe" / "body" / "mediapipe_body_to_standard_human_mapping.yaml"
MEDIAPIPE_HAND_MAPPING = Path(__file__).parent.parent / "detectors" / "keypoint_detectors" / "mediapipe" / "hands" / "mediapipe_hand_to_standard_human_mapping.yaml"

__all__ = [
    "RTMPOSE_BODY_MAPPING",
    "RTMPOSE_HAND_MAPPING",
    "MEDIAPIPE_BODY_MAPPING",
    "MEDIAPIPE_HAND_MAPPING",
]
