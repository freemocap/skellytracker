from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from skellytracker.core.config.session_config import SessionConfig


@dataclass
class Session(ABC):
    """Manages computational resources (GPU/CPU) for a set of detectors.

    One Session per backend per Tracker. Created once, shared across all
    detectors that use that backend. Detectors do not own resource lifecycle.

    `kind` identifies the session's backend WITHOUT importing the concrete
    class, so code that only needs to ask "is this an ONNX session?" never
    pays for onnxruntime's module-level import.
    """

    kind: ClassVar[str]

    @classmethod
    @abstractmethod
    def create(cls, config: SessionConfig) -> Session:
        """Allocate resources: load models, select device, run warmup."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources held by this session."""
        ...
