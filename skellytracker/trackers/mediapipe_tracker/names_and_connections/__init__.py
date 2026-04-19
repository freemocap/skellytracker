"""
Module-level access to the mediapipe tracker definitions.

Definitions are loaded once from YAML on first import and cached. Detectors,
observations, and annotators read from these constants rather than defining
names/connections in Python.
"""

from pathlib import Path

from skellytracker.trackers.base_tracker.tracked_object_definition import TrackedObjectDefinition

_YAML_DIR = Path(__file__).parent

MEDIAPIPE_BODY_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(
    _YAML_DIR / "body" / "mediapipe_body.yaml"
)
MEDIAPIPE_HAND_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(
    _YAML_DIR / "hands" / "mediapipe_hand.yaml"
)
MEDIAPIPE_FACE_CONTOUR_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(
    _YAML_DIR / "face" / "face_contour" / "mediapipe_face_contour.yaml"
)
MEDIAPIPE_FACE_TESSELATED_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(
    _YAML_DIR / "face" / "face_tesselated" / "mediapipe_face_tesselated.yaml"
)
MEDIAPIPE_HOLISTIC_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(
    _YAML_DIR / "mediapipe_holistic.yaml"
)

__all__ = [
    "MEDIAPIPE_BODY_DEFINITION",
    "MEDIAPIPE_HAND_DEFINITION",
    "MEDIAPIPE_FACE_CONTOUR_DEFINITION",
    "MEDIAPIPE_FACE_TESSELATED_DEFINITION",
    "MEDIAPIPE_HOLISTIC_DEFINITION",
]
