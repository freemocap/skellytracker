from pathlib import Path

from skellytracker.trackers.base_tracker.tracked_object_definition import TrackedObjectDefinition

_YAML_DIR = Path(__file__).parent
COCO_17_YAML_PATH = _YAML_DIR / "coco_17.yaml"
RT_POSE_DEFINITION = TrackedObjectDefinition.from_yaml(COCO_17_YAML_PATH)
COCO_17_NAMES = RT_POSE_DEFINITION.tracked_points
