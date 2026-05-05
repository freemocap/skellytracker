"""Backward-compatibility shim — re-exports from the unified model registry.

Previously this module defined ``SubModelSpec`` and ``TrackerPreset`` locally.
They now live in ``skellytracker.utilities.gpu_utils.model_registry`` as
``ModelSpec`` and ``TrackerPreset`` respectively.
"""

from skellytracker.core.model_registry import (  # noqa: F401
    ModelSpec as SubModelSpec,
    TrackerPreset,
)
