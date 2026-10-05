"""Build sessions and Trackers directly from a TrackerConfig.

Every ONNX/MediaPipe/CPU Session a config's detector tree needs must
currently be hand-built by the caller and passed into ``Tracker.create``/
``MultiPersonTracker.create`` as a ``sessions`` dict keyed by backend name.
This module owns that wiring in one place: it walks the config tree,
figures out which backends are required (and, for ONNX, which models),
builds exactly those sessions, and hands back a ready-to-use Tracker.

Usage::

    tracker = build_tracker(config)
    # or, with per-backend tuning:
    tracker = build_tracker(config, onnx_overrides={"execution_provider": "cuda"})
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.config.detector_configs import (
    KeypointDetectorConfig,
    ObjectDetectorConfig,
)
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.detectors.detector_base_classes import (
    KEYPOINT_DETECTOR_REGISTRY,
    OBJECT_DETECTOR_REGISTRY,
)
from skellytracker.core.sessions.session import Session
from skellytracker.core.temporal_processing.multi_person_config import (
    MultiPersonTrackingConfig,
)
from skellytracker.core.tracker.multi_person_tracker import MultiPersonTracker
from skellytracker.core.tracker.tracker import Tracker

# OnnxModelSpec instances, not imported eagerly here: onnx_session.py hard-imports
# onnxruntime, an optional extra, so this module only imports it lazily (inside
# build_sessions) once an ONNX-backed detector is actually found in the config tree.


@dataclass
class RequiredSessions:
    """What a config tree needs, collected by `_collect_required_sessions`."""

    needs_mediapipe: bool = False
    needs_cpu: bool = False
    # Keyed by model name so multiple detectors sharing one ONNX model merge
    # into a single entry (one OnnxSession holds all models for a tracker).
    # Values are OnnxModelSpec instances.
    onnx_model_specs: dict[str, Any] = field(default_factory=dict)


def _model_spec_for(
    detector_cls: type, config: ObjectDetectorConfig | KeypointDetectorConfig
) -> Any:
    model_spec_fn = getattr(detector_cls, "model_spec", None)
    if model_spec_fn is None:
        raise TypeError(
            f"{detector_cls.__name__} has session_backend='onnx' but defines no "
            f"model_spec(model_name) classmethod — cannot build its OnnxModelSpec. "
            f"See YoloxPersonDetector.model_spec for the expected convention."
        )
    model_name = getattr(config, "model_name", None)
    if model_name is None:
        raise TypeError(
            f"{type(config).__name__} has session_backend='onnx' but no model_name field."
        )
    return model_spec_fn(model_name)


def _visit_detector_config(
    config: ObjectDetectorConfig | KeypointDetectorConfig, result: RequiredSessions
) -> None:
    if config.session_backend == "mediapipe":
        result.needs_mediapipe = True
    elif config.session_backend == "cpu":
        result.needs_cpu = True
    elif config.session_backend == "onnx":
        detector_cls = OBJECT_DETECTOR_REGISTRY.get(
            config.detector_type
        ) or KEYPOINT_DETECTOR_REGISTRY.get(config.detector_type)
        if detector_cls is None:
            raise KeyError(
                f"No detector registered for type {config.detector_type!r}. "
                f"Registered object types: {list(OBJECT_DETECTOR_REGISTRY)}. "
                f"Registered keypoint types: {list(KEYPOINT_DETECTOR_REGISTRY)}"
            )
        spec = _model_spec_for(detector_cls, config)
        result.onnx_model_specs[spec.name] = spec


def _collect_required_sessions(stages: list[DetectionStageConfig]) -> RequiredSessions:
    result = RequiredSessions()

    def _recurse(stage: DetectionStageConfig) -> None:
        if stage.object_detector is not None:
            _visit_detector_config(stage.object_detector, result)
        for keypoint_config in stage.keypoint_detectors:
            _visit_detector_config(keypoint_config, result)
        for child in stage.children:
            _recurse(child)

    for root in stages:
        _recurse(root)
    return result


def build_sessions(
    config: TrackerConfig,
    *,
    onnx_overrides: dict[str, Any] | None = None,
    mediapipe_overrides: dict[str, Any] | None = None,
    cpu_overrides: dict[str, Any] | None = None,
) -> dict[str, Session]:
    """Walk `config`'s stage tree and build exactly the sessions it needs."""
    required = _collect_required_sessions(config.stages)

    sessions: dict[str, Session] = {}
    if required.needs_mediapipe:
        from skellytracker.core.sessions.mediapipe_session import (
            MediaPipeSession,
            MediaPipeSessionConfig,
        )

        sessions["mediapipe"] = MediaPipeSession.create(
            MediaPipeSessionConfig(**(mediapipe_overrides or {}))
        )
    if required.needs_cpu:
        from skellytracker.core.sessions.cpu_session import CpuSession, CpuSessionConfig

        sessions["cpu"] = CpuSession.create(CpuSessionConfig(**(cpu_overrides or {})))
    if required.onnx_model_specs:
        from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig

        onnx_kwargs: dict[str, Any] = {
            "batch_size": 1,
            "models": list(required.onnx_model_specs.values()),
            **(onnx_overrides or {}),
        }
        sessions["onnx"] = OnnxSession.create(OnnxSessionConfig(**onnx_kwargs))
    return sessions


def build_tracker(
    config: TrackerConfig,
    *,
    onnx_overrides: dict[str, Any] | None = None,
    mediapipe_overrides: dict[str, Any] | None = None,
    cpu_overrides: dict[str, Any] | None = None,
) -> Tracker:
    """Build the sessions `config` needs and return a ready-to-use Tracker."""
    sessions = build_sessions(
        config,
        onnx_overrides=onnx_overrides,
        mediapipe_overrides=mediapipe_overrides,
        cpu_overrides=cpu_overrides,
    )
    return Tracker.create(config, sessions)


def build_multi_person_tracker(
    config: TrackerConfig,
    multi_person_config: MultiPersonTrackingConfig | None = None,
    *,
    onnx_overrides: dict[str, Any] | None = None,
    mediapipe_overrides: dict[str, Any] | None = None,
    cpu_overrides: dict[str, Any] | None = None,
) -> MultiPersonTracker:
    """Build the sessions `config` needs and return a ready-to-use MultiPersonTracker."""
    sessions = build_sessions(
        config,
        onnx_overrides=onnx_overrides,
        mediapipe_overrides=mediapipe_overrides,
        cpu_overrides=cpu_overrides,
    )
    return MultiPersonTracker.create(config, sessions, multi_person_config)
