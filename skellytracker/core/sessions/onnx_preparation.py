"""Worker-executed preparation with explicit, single-owner native resources.

This module neither starts threads nor imports a native runtime. Applications
schedule ``run`` and ``dispose`` away from inference, and poll ``snapshot`` or
``take_ready`` between requests. The builder owns partial-construction cleanup;
after it returns, this job owns the backend until transfer or disposal.

The validator must check actual provider, model/device/precision/shape identity
and warmup. Disk cache presence is not validation. This lifecycle does not yet
implement native construction, cache publication, or pool adoption.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Event, Lock
from time import monotonic
from typing import Generic, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class PreparedBackend(Protocol):
    def close(self) -> None: ...


Backend = TypeVar("Backend", bound=PreparedBackend)


@dataclass(frozen=True, slots=True)
class OnnxPreparationTicket:
    """Owner-issued identity; it is not itself evidence of compatibility.

    The owner binds request_id/generation to its upgradeable lease and identity
    to the complete resolved configuration. Explicit CUDA leases must not issue
    these tickets. Native adoption must independently verify compatibility.
    """

    request_id: str
    generation: int
    identity: str

    def __post_init__(self) -> None:
        if not self.request_id or not self.identity or self.generation < 0:
            raise ValueError("Preparation requires a request, identity and nonnegative generation")


class PreparationPhase(str, Enum):
    QUEUED = "queued"
    BUILDING = "building"
    VALIDATING = "validating"
    READY = "ready"
    STOPPING = "stopping"
    DISPOSING = "disposing"
    CANCELLED = "cancelled"
    TRANSFERRED = "transferred"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PreparationSnapshot:
    ticket: OnnxPreparationTicket
    phase: PreparationPhase
    elapsed_seconds: float
    reason: str | None


class OnnxPreparationJob(Generic[Backend]):
    """One build, one validation, then exactly one transfer or close attempt.

    ``request_stop`` never closes a resource or waits for a worker. When a ready
    job is stopped, its scheduler must enqueue ``dispose``; stopping an executing
    job makes its worker dispose the result at the next safe boundary. No native
    callback is invoked while holding the status lock.
    """

    def __init__(
        self,
        *,
        ticket: OnnxPreparationTicket,
        build: Callable[[], Backend],
        validate: Callable[[Backend], None],
    ) -> None:
        self._ticket = ticket
        self._build = build
        self._validate = validate
        self._lock = Lock()
        self._stop = Event()
        self._phase = PreparationPhase.QUEUED
        self._reason: str | None = None
        self._started: float | None = None
        self._ended: float | None = None
        self._backend: Backend | None = None

    @property
    def ticket(self) -> OnnxPreparationTicket:
        return self._ticket

    def snapshot(self) -> PreparationSnapshot:
        with self._lock:
            elapsed = 0.0 if self._started is None else (
                (self._ended if self._ended is not None else monotonic()) - self._started
            )
            return PreparationSnapshot(self.ticket, self._phase, elapsed, self._reason)

    def request_stop(self) -> None:
        with self._lock:
            if self._phase in (
                PreparationPhase.TRANSFERRED, PreparationPhase.CANCELLED, PreparationPhase.FAILED,
            ):
                return
            self._stop.set()
            if self._phase == PreparationPhase.QUEUED:
                self._phase = PreparationPhase.CANCELLED
                self._ended = monotonic()
            elif self._phase != PreparationPhase.DISPOSING:
                self._phase = PreparationPhase.STOPPING

    def run(self) -> None:
        """Execute once on the scheduler's worker; propagate failures to it."""
        with self._lock:
            if self._phase == PreparationPhase.CANCELLED:
                return
            if self._phase != PreparationPhase.QUEUED:
                raise RuntimeError("Preparation job has already started")
            self._started = monotonic()
            self._phase = PreparationPhase.BUILDING
        backend: Backend | None = None
        try:
            backend = self._build()
            if not self._stop.is_set():
                with self._lock:
                    if not self._stop.is_set():
                        self._phase = PreparationPhase.VALIDATING
                if not self._stop.is_set():
                    self._validate(backend)
            with self._lock:
                self._backend = backend
                backend = None
                stopped = self._stop.is_set()
                self._phase = PreparationPhase.STOPPING if stopped else PreparationPhase.READY
                self._ended = monotonic()
            if stopped:
                self.dispose()
        except BaseException as error:
            # A failed close is still a terminal ownership attempt, never retried.
            if backend is not None:
                try:
                    backend.close()
                except BaseException as close_error:  # noqa: BLE001 - preserve both failures, including interrupts
                    error = BaseExceptionGroup("Preparation and cleanup failed", [error, close_error])
            with self._lock:
                self._phase = PreparationPhase.FAILED
                self._reason = str(error)
                self._ended = monotonic()
            raise error  # noqa: TRY201 - cleanup may have replaced it with an exception group

    def take_ready(self, *, expected_ticket: OnnxPreparationTicket) -> Backend | None:
        """Transfer without waiting or calling native code; never imply adoption.

        A mismatched/stale ticket leaves ownership here for worker disposal.
        The caller must arrange actual adoption and cleanup after transfer.
        """
        with self._lock:
            if expected_ticket != self.ticket:
                raise ValueError("Prepared backend ticket does not match the current request")
            if self._phase != PreparationPhase.READY:
                return None
            backend = self._backend
            self._backend = None
            self._phase = PreparationPhase.TRANSFERRED
            return backend

    def dispose(self) -> None:
        """Close an untransferred ready backend on a cleanup worker, at most once.

        Running constructors cannot be disposed: use request_stop and let run
        reach its boundary. Repeated disposal after termination is harmless.
        """
        with self._lock:
            if self._phase in (
                PreparationPhase.CANCELLED, PreparationPhase.TRANSFERRED,
                PreparationPhase.FAILED, PreparationPhase.DISPOSING,
            ):
                return
            if self._backend is None:
                raise RuntimeError("Request stop instead of disposing an uncompleted build")
            backend = self._backend
            self._backend = None
            self._phase = PreparationPhase.DISPOSING
        try:
            backend.close()
        except BaseException as error:
            with self._lock:
                self._phase = PreparationPhase.FAILED
                self._reason = str(error)
                self._ended = monotonic()
            raise
        with self._lock:
            self._phase = PreparationPhase.CANCELLED
            self._ended = monotonic()
