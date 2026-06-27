"""
Module-level access to RTMO hybrid tracker definitions.

Definitions are loaded once from YAML on first import and cached. Detectors,
observations, and annotators read from these constants rather than defining
names/connections in Python.

The hybrid composition references hand/face YAMLs from rtmpose_tracker via
relative paths — the composed_of mechanism resolves them automatically.
"""

from pathlib import Path

from skellytracker.old.base_tracker.tracked_object_definition import TrackedObjectDefinition

_YAML_DIR = Path(__file__).parent
RTMO_BODY_17_YAML_PATH = _YAML_DIR / "rtmo_body_17.yaml"
RTMO_HYBRID_YAML_PATH = _YAML_DIR / "rtmo_hybrid.yaml"

for yaml_path in (RTMO_BODY_17_YAML_PATH, RTMO_HYBRID_YAML_PATH):
    if not yaml_path.exists():
        raise FileNotFoundError(yaml_path)

RTMO_BODY_17_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(RTMO_BODY_17_YAML_PATH)
RTMO_HYBRID_DEFINITION: TrackedObjectDefinition = TrackedObjectDefinition.from_yaml(RTMO_HYBRID_YAML_PATH)

__all__ = [
    "RTMO_BODY_17_DEFINITION",
    "RTMO_HYBRID_DEFINITION",
    "RTMO_BODY_17_YAML_PATH",
    "RTMO_HYBRID_YAML_PATH",
]
