"""Reference-counted session ownership for independently stateful trackers."""

from dataclasses import dataclass, field
from threading import get_ident

from skellytracker.core.config.session_config import SessionConfig
from skellytracker.core.config.tracker_config import TrackerConfig
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


@dataclass(eq=False, slots=True)
class TrackerLease:
    """One independent tracker and its references to pooled sessions."""

    tracker: Tracker
    _pool: "SharedSessionPool"
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

    def create_tracker(self, *, config: TrackerConfig, requests: tuple[SessionRequest, ...]) -> TrackerLease:
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
                entry = next((item for item in self._entries if item.request == request), None)
                if entry is None:
                    snapshot = SessionRequest(
                        session_type=request.session_type, config=request.config.model_copy(deep=True),
                    )
                    entry = SessionEntry(request=snapshot, session=snapshot.session_type.create(snapshot.config))
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

    def _release_entries(self, *, entries: tuple[SessionEntry, ...]) -> None:
        errors: list[Exception] = []
        for entry in entries:
            entry.references -= 1
            if entry.references == 0:
                self._entries.remove(entry)
                try:
                    entry.session.close()
                except Exception as error:
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
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Failed to close shared trackers", errors)
