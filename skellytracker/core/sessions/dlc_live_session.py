from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from skellytracker.core.config.session_config import SessionConfig
from skellytracker.core.sessions.session import Session


class DLCLiveSessionConfig(SessionConfig):
    backend: Literal["dlclive"] = "dlclive"
    device: str | None = None
    precision: Literal["FP32", "FP16"] = "FP32"


@dataclass
class DLCLiveSession(Session):
    """Shared device/precision context for DeepLabCut-Live (PyTorch) detectors.

    dlclive.DLCLive objects load and own a model per instance and are not
    shared/thread-safe across cameras, so unlike OnnxSession this session
    holds no model state — each DLCLiveKeypointDetector creates and owns its
    own DLCLive instance (mirrors MediaPipeSession).
    """

    device: str | None
    precision: Literal["FP32", "FP16"]

    @classmethod
    def create(cls, config: DLCLiveSessionConfig | None = None) -> DLCLiveSession:
        if config is None:
            config = DLCLiveSessionConfig()
        return cls(device=config.device, precision=config.precision)

    def close(self) -> None:
        pass
