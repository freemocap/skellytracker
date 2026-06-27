"""Legacy mediapipe tracker package.

Heavy classes are exposed lazily via module ``__getattr__`` (PEP 562) so that
importing the lightweight ``LegacyMediapipeDetectorConfig`` — or any sibling
submodule — does not pull the ``mediapipe`` native library into the import
graph. See ``mediapipe_tracker/__init__.py`` for the rationale.
"""
import importlib
from typing import TYPE_CHECKING

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "LegacyMediapipeTracker": (".__legacy_mediapipe_tracker", "LegacyMediapipeTracker"),
    "LegacyMediapipeDetectorConfig": (".legacy_mediapipe_detector_config", "LegacyMediapipeDetectorConfig"),
    "LegacyMediapipeObservation": (".legacy_mediapipe_observation", "LegacyMediapipeObservation"),
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name: str):
    try:
        module_rel, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_rel, __name__), attr)
    globals()[name] = value
    return value


if TYPE_CHECKING:  # static-tooling visibility; resolved lazily at runtime above
    from skellytracker.old.legacy_mediapipe_tracker.__legacy_mediapipe_tracker import LegacyMediapipeTracker
    from skellytracker.old.legacy_mediapipe_tracker.legacy_mediapipe_detector_config import LegacyMediapipeDetectorConfig
    from skellytracker.old.legacy_mediapipe_tracker.legacy_mediapipe_observation import LegacyMediapipeObservation
