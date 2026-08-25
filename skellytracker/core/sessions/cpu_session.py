from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from skellytracker.core.config.session_config import SessionConfig
from skellytracker.core.sessions.session import Session


class CpuSessionConfig(SessionConfig):
    backend: Literal["cpu"] = "cpu"


@dataclass
class CpuSession(Session):
    """No-op session for pure-CPU/OpenCV detectors that need no model loading."""

    kind: ClassVar[str] = "cpu"

    @classmethod
    def create(cls, config: CpuSessionConfig) -> CpuSession:
        return cls()

    def close(self) -> None:
        pass
