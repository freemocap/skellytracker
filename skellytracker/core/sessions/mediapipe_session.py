from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from skellytracker.core.config.session_config import SessionConfig
from skellytracker.core.sessions.session import Session
from skellytracker.core.detectors.keypoint_detectors.mediapipe.mediapipe_model_manager import (
    MediapipePoseModelComplexity,
    get_pose_model_path,
)

logger = logging.getLogger(__name__)

_GPU_ERROR_MARKERS = ("kGpuService", "NSOpenGLContext", "GlContext", "EGL")


class MediaPipeSessionConfig(SessionConfig):
    backend: Literal["mediapipe"] = "mediapipe"
    running_mode: Literal["video", "image"] = "video"
    disable_gpu: bool | None = None  # None = auto-detect


def _is_gpu_error(exc: RuntimeError) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _GPU_ERROR_MARKERS)


@dataclass
class MediaPipeSession(Session):
    """Manages the shared MediaPipe device context (GPU/CPU selection).

    Performs a one-time GPU probe at creation and sets the MEDIAPIPE_DISABLE_GPU
    environment variable before any landmarkers are created. Each detector then
    creates and owns its own landmarker using this device context.
    """

    kind: ClassVar[str] = "mediapipe"

    running_mode: Literal["video", "image"]

    @classmethod
    def create(cls, config: MediaPipeSessionConfig | None = None) -> MediaPipeSession:
        if config is None:
            config = MediaPipeSessionConfig()
        # Suppress noisy-but-harmless C++ glog warnings from MediaPipe's internal
        # graph (e.g. landmark_projection_calculator NORM_RECT warning on non-square images).
        os.environ.setdefault("GLOG_minloglevel", "2")

        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        mp_running_mode = RunningMode.VIDEO if config.running_mode == "video" else RunningMode.IMAGE

        disable_gpu = config.disable_gpu
        if disable_gpu is None:
            disable_gpu = _probe_gpu(
                BaseOptions=BaseOptions,
                PoseLandmarker=PoseLandmarker,
                PoseLandmarkerOptions=PoseLandmarkerOptions,
                mp_running_mode=mp_running_mode,
            )

        if disable_gpu:
            os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
        else:
            os.environ.pop("MEDIAPIPE_DISABLE_GPU", None)

        return cls(running_mode=config.running_mode)

    def close(self) -> None:
        pass


def _probe_gpu(
    BaseOptions: Any,
    PoseLandmarker: Any,
    PoseLandmarkerOptions: Any,
    mp_running_mode: Any,
) -> bool:
    """Probe whether MediaPipe GPU context creation works in this environment.

    Always uses the lite pose model to keep the probe download small.
    Returns True if GPU should be disabled, False if GPU is available.
    """
    probe_path = get_pose_model_path(MediapipePoseModelComplexity.LITE)
    os.environ.pop("MEDIAPIPE_DISABLE_GPU", None)
    probe_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(probe_path)),
        running_mode=mp_running_mode,
        num_poses=1,
    )
    try:
        lm = PoseLandmarker.create_from_options(probe_opts)
        lm.close()
        logger.debug("MediaPipe GPU probe succeeded — using GPU mode.")
        return False
    except RuntimeError as exc:
        if _is_gpu_error(exc):
            logger.info(
                "MediaPipe GPU context unavailable (%s); falling back to CPU mode.",
                type(exc).__name__,
            )
            return True
        raise
