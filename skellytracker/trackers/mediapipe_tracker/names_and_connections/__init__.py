"""
Module-level access to the mediapipe tracker definitions.

Definitions are loaded once from YAML on first import and cached. Detectors,
observations, and annotators read from these constants rather than defining
names/connections in Python.
"""

from pathlib import Path

from skellytracker.trackers.base_tracker.tracked_object_definition import TrackedObjectDefinition

_YAML_DIR = Path(__file__).parent
MEDIAPIPE_BODY_YAML_PATH= _YAML_DIR / "mediapipe_body.yaml"
MEDIAPIPE_HAND_YAML_PATH= _YAML_DIR / "mediapipe_hand.yaml"
MEDIAPIPE_FACE_CONTOUR_YAML_PATH= _YAML_DIR / "mediapipe_face_contour.yaml"
MEDIAPIPE_FACE_TESSELATED_YAML_PATH= _YAML_DIR / "mediapipe_face_tesselated.yaml"
MEDIAPIPE_HOLISTIC_YAML_PATH= _YAML_DIR / "mediapipe_holistic.yaml"

for yaml_path in (
        MEDIAPIPE_BODY_YAML_PATH,
        MEDIAPIPE_HAND_YAML_PATH,
        MEDIAPIPE_FACE_CONTOUR_YAML_PATH,
        MEDIAPIPE_FACE_TESSELATED_YAML_PATH,
        MEDIAPIPE_HOLISTIC_YAML_PATH,):
    if not yaml_path.exists():
        raise FileNotFoundError(yaml_path)

MEDIAPIPE_BODY_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(MEDIAPIPE_BODY_YAML_PATH)
MEDIAPIPE_HAND_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(MEDIAPIPE_HAND_YAML_PATH)
MEDIAPIPE_FACE_CONTOUR_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(MEDIAPIPE_FACE_CONTOUR_YAML_PATH)
MEDIAPIPE_FACE_TESSELATED_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(MEDIAPIPE_FACE_TESSELATED_YAML_PATH)
MEDIAPIPE_HOLISTIC_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(MEDIAPIPE_HOLISTIC_YAML_PATH)

__all__ = [
    "MEDIAPIPE_BODY_DEFINITION",
    "MEDIAPIPE_HAND_DEFINITION",
    "MEDIAPIPE_FACE_CONTOUR_DEFINITION",
    "MEDIAPIPE_FACE_TESSELATED_DEFINITION",
    "MEDIAPIPE_HOLISTIC_DEFINITION",
    "MEDIAPIPE_BODY_YAML_PATH",
    "MEDIAPIPE_HAND_YAML_PATH",
    "MEDIAPIPE_FACE_CONTOUR_YAML_PATH",
    "MEDIAPIPE_FACE_TESSELATED_YAML_PATH",
    "MEDIAPIPE_HOLISTIC_YAML_PATH",
]
