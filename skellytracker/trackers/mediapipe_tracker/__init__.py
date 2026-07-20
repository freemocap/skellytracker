"""Mediapipe tracker package.

The heavy detector/tracker/observation/annotator classes are exposed lazily via
module ``__getattr__`` (PEP 562). Importing this package — or any sibling
submodule under it, since Python runs this ``__init__`` first — therefore does
NOT import the ``mediapipe`` native library. mediapipe is only loaded when a
heavy attribute is actually accessed (e.g. ``MediapipeDetector``). Lightweight
config classes (e.g. ``MediapipeDetectorConfig``) resolve to backend-free
modules, so building the detector-config union stays cheap.

This matters because importing a detector *config* used to transitively load
mediapipe, which then corrupts the process when it coexists with other heavy
native libs (the calibration solvers + pyarrow) — a hard segfault.
"""
import importlib
from typing import TYPE_CHECKING

# name -> (relative module, attribute in that module)
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "MediapipeTracker": (".__mediapipe_tracker", "MediapipeCompositeTracker"),
    "MediapipePoseTracker": (".body.mediapipe_pose_tracker", "MediapipePoseTracker"),
    "MediapipeAnnotator": (".composite.mediapipe_composite_annotator", "MediapipeCompositeAnnotator"),
    "MediapipeAnnotatorConfig": (".composite.mediapipe_composite_annotator", "MediapipeCompositeAnnotatorConfig"),
    "MediapipeDetectorConfig": (".composite.mediapipe_composite_config", "MediapipeCompositeDetectorConfig"),
    "MediapipeDetector": (".composite.mediapipe_composite_detector", "MediapipeCompositeDetector"),
    "MediapipeObservation": (".composite.mediapipe_composite_observation", "MediapipeCompositeObservation"),
    "MediapipeTrackerConfig": (".composite.mediapipe_composite_tracker_config", "MediapipeCompositeTrackerConfig"),
    "MediapipeFaceTracker": (".face.mediapipe_face_tracker", "MediapipeFaceTracker"),
    "MediapipeHandTracker": (".hands.mediapipe_hand_tracker", "MediapipeHandTracker"),
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name: str):
    try:
        module_rel, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_rel, __name__), attr)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


if TYPE_CHECKING:  # static-tooling visibility; resolved lazily at runtime above
    from skellytracker.trackers.mediapipe_tracker.__mediapipe_tracker import MediapipeCompositeTracker as MediapipeTracker
    from skellytracker.trackers.mediapipe_tracker.body.mediapipe_pose_tracker import MediapipePoseTracker
    from skellytracker.trackers.mediapipe_tracker.composite.mediapipe_composite_annotator import (
        MediapipeCompositeAnnotator as MediapipeAnnotator,
        MediapipeCompositeAnnotatorConfig as MediapipeAnnotatorConfig,
    )
    from skellytracker.trackers.mediapipe_tracker.composite.mediapipe_composite_config import MediapipeCompositeDetectorConfig as MediapipeDetectorConfig
    from skellytracker.trackers.mediapipe_tracker.composite.mediapipe_composite_detector import MediapipeCompositeDetector as MediapipeDetector
    from skellytracker.trackers.mediapipe_tracker.composite.mediapipe_composite_observation import MediapipeCompositeObservation as MediapipeObservation
    from skellytracker.trackers.mediapipe_tracker.composite.mediapipe_composite_tracker_config import MediapipeCompositeTrackerConfig as MediapipeTrackerConfig
    from skellytracker.trackers.mediapipe_tracker.face.mediapipe_face_tracker import MediapipeFaceTracker
    from skellytracker.trackers.mediapipe_tracker.hands.mediapipe_hand_tracker import MediapipeHandTracker
