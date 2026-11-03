"""Reference-counted session ownership for independently stateful trackers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import get_ident
from uuid import uuid4

from skellytracker.core.config.session_config import SessionConfig
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.sessions.onnx_preparation import (
    OnnxPreparationJob,
    OnnxPreparationTicket,
    PreparationPhase,
)
from skellytracker.core.sessions.session import Session
from skellytracker.core.tracker.tracker import Tracker


@dataclass(frozen=True, slots=True)
class SessionRequest:
    session_type: type[Session]
    config: SessionConfig


@dataclass(slots=True)
class SessionEntry:
    request: SessionRequest
    session: Session
    references: int = 0
    upgradeable: bool = False
    preparation: OnnxPreparationJob | None = None
    activation: BackendActivation | None = None
    generation: int = 0


@dataclass(frozen=True, slots=True)
class BackendActivation:
    cleanup: Session | None = None
    activated: bool = False
    error: str | None = None


@dataclass(eq=False, slots=True)
class TrackerLease:
    """One independent tracker and its references to pooled sessions."""

    tracker: Tracker
    _pool: SharedSessionPool
    _entries: tuple[SessionEntry, ...]
    _closed: bool = False

    def close(self) -> None:
        self._pool.release(lease=self)


@dataclass
class SharedSessionPool:
    """Own sessions on the inference execution thread.

    Exact configuration equality is required for reuse, including batch size,
    device, provider, model descriptors and precision. No mutable detector is
    pooled. Sessions are released when their last tracker closes, so idle model
    resources do not accumulate. This pool does not schedule inference calls.
    """

    _owner: int = field(default_factory=get_ident, init=False)
    _entries: list[SessionEntry] = field(default_factory=list, init=False)
    _leases: list[TrackerLease] = field(default_factory=list, init=False)
    _closed: bool = field(default=False, init=False)

    def _require_owner(self) -> None:
        if get_ident() != self._owner:
            raise RuntimeError("SharedSessionPool must be used on its owning inference thread")

    def create_tracker(self, *, config: TrackerConfig, requests: tuple[SessionRequest, ...],
                       cuda_first: bool = False, progress: Callable[[str, str], None] | None = None) -> TrackerLease:
        self._require_owner()
        if self._closed:
            raise RuntimeError("SharedSessionPool is closed")
        backends = [request.config.backend for request in requests]
        if len(backends) != len(set(backends)):
            raise ValueError("Only one session request per backend is allowed")
        acquired: list[SessionEntry] = []
        try:
            for request in requests:
                if request.config.backend != request.session_type.kind:
                    raise ValueError("Session configuration backend does not match session type")
                upgradeable = (cuda_first and request.config.backend == "onnx"
                               and request.config.execution_provider in (None, "trt"))
                if upgradeable:
                    from skellytracker.core.tracker.tracker_factory import (
                        collect_required_sessions,
                    )
                    initial = request.config.model_copy(deep=True)
                    proposed = request.config.model_copy(deep=True)
                    initial.execution_provider = "cuda"
                    proposed.execution_provider = "trt"
                    validators = collect_required_sessions(config).onnx_validators
                    for validator in validators:
                        validator(initial)
                    try:
                        for validator in validators:
                            validator(proposed)
                    except (ValueError, RuntimeError):
                        upgradeable = False
                    request = SessionRequest(session_type=request.session_type, config=initial)
                entry = next((item for item in self._entries
                              if item.request == request and item.upgradeable == upgradeable), None)
                if entry is None:
                    snapshot = SessionRequest(
                        session_type=request.session_type, config=request.config.model_copy(deep=True),
                    )
                    initial = snapshot.config.model_copy(deep=True)
                    if upgradeable:
                        initial.execution_provider = "cuda"
                    options = {"progress": progress} if progress is not None and initial.backend == "onnx" else {}
                    entry = SessionEntry(request=snapshot, session=snapshot.session_type.create(initial, **options),
                                         upgradeable=upgradeable)
                    self._entries.append(entry)
                entry.references += 1
                acquired.append(entry)
            tracker = Tracker.create_with_shared_sessions(
                config=config,
                sessions={entry.request.config.backend: entry.session for entry in acquired},
            )
        except BaseException:
            self._release_entries(entries=tuple(acquired))
            raise
        lease = TrackerLease(tracker=tracker, _pool=self, _entries=tuple(acquired))
        self._leases.append(lease)
        return lease

    def can_prepare_onnx(self, *, lease: TrackerLease) -> bool:
        self._require_owner()
        if lease._pool is not self or lease._closed:
            return False
        return any(entry.upgradeable and entry.session.execution_provider == "cuda" for entry in lease._entries)

    def onnx_provider(self, *, lease: TrackerLease) -> str | None:
        """Read owner-thread provider metadata, including shared entry upgrades."""
        self._require_owner()
        if lease._pool is not self or lease._closed:
            raise ValueError("Tracker lease is no longer owned by this pool")
        for entry in lease._entries:
            if entry.request.config.backend == "onnx":
                return entry.session.execution_provider
        return None

    def prepare_onnx(self, *, lease: TrackerLease, isolated: bool = True,
                     retry: bool = False) -> OnnxPreparationJob | None:
        """Issue/coalesce a job after CUDA inference, without running native work."""
        self._require_owner()
        if lease._pool is not self or lease._closed:
            raise ValueError("Preparation lease is no longer owned by this pool")
        for entry in lease._entries:
            if not entry.upgradeable:
                continue
            if retry and entry.preparation is not None:
                terminal = entry.preparation.snapshot().phase in (PreparationPhase.FAILED, PreparationPhase.CANCELLED)
                terminal = terminal or (entry.activation is not None and entry.activation.error is not None)
                if terminal:
                    runtime = getattr(entry.preparation, "runtime", None)
                    if runtime is not None and runtime.status().alive:
                        raise ValueError("Previous native worker is still stopping")
                    entry.preparation = None
                    entry.activation = None
                    entry.generation += 1
            if entry.preparation is None:
                ticket = OnnxPreparationTicket(str(uuid4()), entry.generation, str(uuid4()))
                factory = entry.session.isolated_preparation_job if isolated else entry.session.preparation_job
                entry.preparation = factory(ticket)
            return entry.preparation
        return None

    def activate_prepared(self, *, lease: TrackerLease, job: OnnxPreparationJob) -> BackendActivation:
        """Between requests only; return resources for off-thread disposal."""
        self._require_owner()
        if lease._pool is not self or lease._closed:
            raise ValueError("Preparation lease is no longer owned by this pool")
        entry = next((item for item in lease._entries if item.preparation is job and item.upgradeable), None)
        if entry is None:
            raise ValueError("Preparation does not belong to the current session generation")
        candidate = job.take_ready(expected_ticket=job.ticket)
        if candidate is None:
            return entry.activation or BackendActivation()
        try:
            retired = entry.session.adopt_prepared_backend(candidate)
        except Exception as error:  # noqa: BLE001 - return failed candidate for off-thread cleanup
            entry.activation = BackendActivation(error=str(error))
            return BackendActivation(cleanup=candidate, error=str(error))
        entry.activation = BackendActivation(activated=True)
        return BackendActivation(cleanup=retired, activated=True)

    def _release_entries(self, *, entries: tuple[SessionEntry, ...]) -> None:
        errors: list[Exception] = []
        for entry in entries:
            entry.references -= 1
            if entry.references == 0:
                self._entries.remove(entry)
                if entry.preparation is not None:
                    entry.preparation.request_stop()
                try:
                    entry.session.close()
                except Exception as error:  # noqa: BLE001 - aggregate all resource cleanup failures
                    errors.append(error)
        if errors:
            raise ExceptionGroup("Failed to close inference sessions", errors)

    def release(self, *, lease: TrackerLease) -> None:
        self._require_owner()
        if lease._pool is not self:
            raise ValueError("Tracker lease belongs to another session pool")
        if lease._closed:
            return
        lease._closed = True
        self._leases.remove(lease)
        try:
            lease.tracker.close()
        finally:
            self._release_entries(entries=lease._entries)

    def close(self) -> None:
        self._require_owner()
        self._closed = True
        errors: list[Exception] = []
        for lease in tuple(self._leases):
            try:
                lease.close()
            except Exception as error:  # noqa: BLE001 - aggregate all lease cleanup failures
                errors.append(error)
        if errors:
            raise ExceptionGroup("Failed to close shared trackers", errors)
