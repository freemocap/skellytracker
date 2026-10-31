"""Registry-based construction adapted from PR 90 [@skellytrackerFactoryIntegration].

Discover model requirements once. Callers can allocate owned sessions or pass the
same requests to SharedSessionPool without allocating duplicate sessions.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.detectors.detector_base_classes import (
    KEYPOINT_DETECTOR_REGISTRY,
    OBJECT_DETECTOR_REGISTRY,
)
from skellytracker.core.sessions.onnx_model_spec import OnnxModelSpec
from skellytracker.core.sessions.session import Session
from skellytracker.core.sessions.shared_sessions import SessionRequest
from skellytracker.core.tracker.tracker import Tracker
from skellytracker.core.tracker.multi_person_tracker import MultiPersonTracker
from skellytracker.core.temporal_processing.multi_person_config import (
    MultiPersonTrackingConfig,
)


@dataclass
class RequiredSessions:
    backends: set[str] = field(default_factory=set)
    onnx_model_specs: dict[str, OnnxModelSpec] = field(default_factory=dict)

    @property
    def needs_mediapipe(self) -> bool:
        return "mediapipe" in self.backends

    @property
    def needs_cpu(self) -> bool:
        return "cpu" in self.backends


def collect_required_sessions(config: TrackerConfig) -> RequiredSessions:
    """Describe registered detectors throughout the stage tree without loading models."""
    result = RequiredSessions()
    pending = list(config.stages)
    while pending:
        stage = pending.pop()
        pending.extend(stage.children)
        detectors = [
            (detector, KEYPOINT_DETECTOR_REGISTRY)
            for detector in stage.keypoint_detectors
        ]
        if stage.object_detector is not None:
            detectors.append((stage.object_detector, OBJECT_DETECTOR_REGISTRY))
        for detector, registry in detectors:
            detector_type = registry.get(detector.detector_type)
            if detector_type is None:
                raise KeyError(f"Unregistered detector: {detector.detector_type!r}")
            backend = detector.session_backend
            if backend not in ("onnx", "mediapipe", "cpu"):
                raise ValueError(f"Unsupported session backend: {backend!r}")
            result.backends.add(backend)
            if backend == "onnx":
                model_spec = getattr(detector_type, "model_spec", None)
                model_name = getattr(detector, "model_name", None)
                if not callable(model_spec) or model_name is None:
                    raise TypeError(
                        f"ONNX detector {detector.detector_type!r} requires model_name and model_spec"
                    )
                descriptor = model_spec(model_name)
                if not isinstance(descriptor, OnnxModelSpec):
                    raise TypeError(
                        f"Invalid model descriptor for {detector.detector_type!r}"
                    )
                previous = result.onnx_model_specs.get(descriptor.name)
                if previous is not None and previous != descriptor:
                    raise ValueError(
                        f"Conflicting descriptors use model name {descriptor.name!r}"
                    )
                result.onnx_model_specs[descriptor.name] = descriptor
    return result


def tracker_session_requests(
    config: TrackerConfig,
    *,
    batch_size: int = 1,
    onnx_overrides: dict[str, Any] | None = None,
    mediapipe_overrides: dict[str, Any] | None = None,
    cpu_overrides: dict[str, Any] | None = None,
) -> tuple[SessionRequest, ...]:
    """Describe requests for the shared-session pool without creating sessions."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    requirements = collect_required_sessions(config)
    requests = []
    for backend in sorted(requirements.backends):
        if backend == "cpu":
            from skellytracker.core.sessions.cpu_session import (
                CpuSession,
                CpuSessionConfig,
            )

            requests.append(
                SessionRequest(
                    session_type=CpuSession,
                    config=CpuSessionConfig(**(cpu_overrides or {})),
                )
            )
        elif backend == "mediapipe":
            from skellytracker.core.sessions.mediapipe_session import (
                MediaPipeSession,
                MediaPipeSessionConfig,
            )

            requests.append(
                SessionRequest(
                    session_type=MediaPipeSession,
                    config=MediaPipeSessionConfig(**(mediapipe_overrides or {})),
                )
            )
        else:
            from skellytracker.core.sessions.onnx_session import OnnxSession
            from skellytracker.core.sessions.onnx_model_spec import OnnxSessionConfig

            overrides = dict(onnx_overrides or {})
            if "models" in overrides:
                raise ValueError(
                    "ONNX model requirements come from detector configuration"
                )
            values = dict(batch_size=batch_size)
            values.update(overrides)
            values["models"] = [
                requirements.onnx_model_specs[name]
                for name in sorted(requirements.onnx_model_specs)
            ]
            requests.append(
                SessionRequest(
                    session_type=OnnxSession, config=OnnxSessionConfig(**values)
                )
            )
    return tuple(requests)


def build_sessions(
    config: TrackerConfig,
    *,
    batch_size: int = 1,
    onnx_overrides: dict[str, Any] | None = None,
    mediapipe_overrides: dict[str, Any] | None = None,
    cpu_overrides: dict[str, Any] | None = None,
) -> dict[str, Session]:
    requests = tracker_session_requests(
        config,
        batch_size=batch_size,
        onnx_overrides=onnx_overrides,
        mediapipe_overrides=mediapipe_overrides,
        cpu_overrides=cpu_overrides,
    )
    with ExitStack() as cleanup:
        sessions = {}
        for request in requests:
            session = request.session_type.create(request.config)
            cleanup.callback(session.close)
            sessions[request.config.backend] = session
        cleanup.pop_all()
    return sessions


def build_tracker(
    config: TrackerConfig,
    *,
    batch_size: int = 1,
    onnx_overrides: dict[str, Any] | None = None,
    mediapipe_overrides: dict[str, Any] | None = None,
    cpu_overrides: dict[str, Any] | None = None,
) -> Tracker:
    sessions = build_sessions(
        config,
        batch_size=batch_size,
        onnx_overrides=onnx_overrides,
        mediapipe_overrides=mediapipe_overrides,
        cpu_overrides=cpu_overrides,
    )
    with ExitStack() as cleanup:
        for session in sessions.values():
            cleanup.callback(session.close)
        tracker = Tracker.create(config, sessions)
        cleanup.pop_all()
    return tracker


def build_multi_person_tracker(
    config: TrackerConfig,
    multi_person_config: MultiPersonTrackingConfig | None = None,
    *,
    batch_size: int = 1,
    onnx_overrides: dict[str, Any] | None = None,
    mediapipe_overrides: dict[str, Any] | None = None,
    cpu_overrides: dict[str, Any] | None = None,
) -> MultiPersonTracker:
    sessions = build_sessions(
        config,
        batch_size=batch_size,
        onnx_overrides=onnx_overrides,
        mediapipe_overrides=mediapipe_overrides,
        cpu_overrides=cpu_overrides,
    )
    with ExitStack() as cleanup:
        for session in sessions.values():
            cleanup.callback(session.close)
        tracker = MultiPersonTracker.create(config, sessions, multi_person_config)
        cleanup.pop_all()
    return tracker
