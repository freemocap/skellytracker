"""Backward-compatibility re-export shim.

``ModelSpec``, ``ModelSource``, ``TrackerPreset``, ``MODEL_URLS``, and
``resolve_model_path`` have moved to ``skellytracker.core.model_registry``.
Import from there directly in new code.
"""

from skellytracker.core.model_registry import (  # noqa: F401
    MODEL_URLS,
    DEFAULT_CACHE_DIR,
    ModelSource,
    ModelSpec,
    TrackerPreset,
    resolve_model_path,
)
