"""
Module-level access to the RTMPose tracker definitions.

Definitions are loaded once from YAML on first import and cached. Detectors,
observations, and annotators read from these constants rather than defining
names/connections in Python.
"""

from pathlib import Path

from skellytracker.trackers.base_tracker.tracked_object_definition import TrackedObjectDefinition

_YAML_DIR = Path(__file__).parent
RTMPOSE_BODY_YAML_PATH = _YAML_DIR / "rtmpose_body.yaml"
RTMPOSE_HAND_YAML_PATH = _YAML_DIR / "rtmpose_hand.yaml"
RTMPOSE_FACE_YAML_PATH = _YAML_DIR / "rtmpose_face.yaml"
RTMPOSE_WHOLEBODY_YAML_PATH = _YAML_DIR / "rtmpose_wholebody.yaml"

for yaml_path in (
        RTMPOSE_BODY_YAML_PATH,
        RTMPOSE_HAND_YAML_PATH,
        RTMPOSE_FACE_YAML_PATH,
        RTMPOSE_WHOLEBODY_YAML_PATH,):
    if not yaml_path.exists():
        raise FileNotFoundError(yaml_path)

RTMPOSE_BODY_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(RTMPOSE_BODY_YAML_PATH)
RTMPOSE_HAND_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(RTMPOSE_HAND_YAML_PATH)
RTMPOSE_FACE_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(RTMPOSE_FACE_YAML_PATH)
RTMPOSE_WHOLEBODY_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(RTMPOSE_WHOLEBODY_YAML_PATH)

__all__ = [
    "RTMPOSE_BODY_DEFINITION",
    "RTMPOSE_HAND_DEFINITION",
    "RTMPOSE_FACE_DEFINITION",
    "RTMPOSE_WHOLEBODY_DEFINITION",
    "RTMPOSE_BODY_YAML_PATH",
    "RTMPOSE_HAND_YAML_PATH",
    "RTMPOSE_FACE_YAML_PATH",
    "RTMPOSE_WHOLEBODY_YAML_PATH",
]
