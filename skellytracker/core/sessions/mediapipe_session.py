from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import ClassVar, Literal

from skellytracker.core.config.session_config import SessionConfig
from skellytracker.core.sessions.session import Session

logger = logging.getLogger(__name__)


class MediaPipeSessionConfig(SessionConfig):
    backend: Literal["mediapipe"] = "mediapipe"
    running_mode: Literal["video", "image"] = "video"
    disable_gpu: bool | None = None  # None and True select the supported CPU backend.


@dataclass
class MediaPipeSession(Session):
    """Shared CPU task configuration; individual detectors own their task handles.

    ONNX execution-provider settings do not control MediaPipe Tasks. Each detector
    explicitly requests the CPU delegate. Native GPU execution is not implemented.
    """

    kind: ClassVar[str] = "mediapipe"
    running_mode: Literal["video", "image"]

    @classmethod
    def create(cls, config: MediaPipeSessionConfig | None = None) -> MediaPipeSession:
        if config is None:
            config = MediaPipeSessionConfig()
        if config.disable_gpu is False:
            raise ValueError(
                "MediaPipe GPU execution is not implemented; use disable_gpu=True "
                "or the default CPU configuration."
            )
        # Suppress noisy C++ graph diagnostics, without changing delegate selection.
        os.environ.setdefault("GLOG_minloglevel", "2")
        logger.info("MediaPipe tasks use the CPU delegate.")
        return cls(running_mode=config.running_mode)

    def close(self) -> None:
        pass
