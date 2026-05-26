"""Hot-swappable Rust backend for CompositeGPU tracker.

Phase 3: Full Rust implementation with CUDA GPU inference across three
ONNX models (RTMO body + MediaPipe hand + RTMPose face), producing a
165-point PointCloud.

- ``USE_RUST_BACKEND = True`` selects the Rust PyO3 bridge
- ``USE_RUST_BACKEND = False`` falls back to Python ``CompositeGPUTracker``
- ``get_composite_gpu_tracker()`` is the single factory function
"""

import logging
import os
import platform
from typing import Any

import numpy as np

from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseTracker,
    BaseRecorder,
)
from skellytracker.trackers.composite_gpu_tracker.__composite_gpu_tracker import (
    CompositeGPUTracker,
    CompositeGPURecorder,
)
from skellytracker.trackers.composite_gpu_tracker.composite_gpu_config import (
    CompositeGPUTrackerConfig,
    CompositeGPUDetectorConfig,
)
from skellytracker.trackers.composite_gpu_tracker.composite_gpu_detector import (
    CompositeGPUDetector,
)
from skellytracker.trackers.composite_gpu_tracker.composite_gpu_annotator import (
    CompositeGPUImageAnnotator,
    CompositeGPUImageAnnotatorConfig,
)

logger = logging.getLogger(__name__)

# -- Backend selector ------------------------------------------------------------
USE_RUST_BACKEND: bool = True

from skellytracker.trackers._opencv_setup import setup as _setup_opencv
_setup_opencv()

# -- NVIDIA DLL discovery -------------------------------------------------------

try:
    from skellytracker.trackers.rtmpose_tracker.rust_bridge import _setup_nvidia_dlls, _preload_ort_dll
    _setup_nvidia_dlls()
    _preload_ort_dll()
except ImportError:
    pass

# -- Lazy import ---------------------------------------------------------------

_native_module: Any = None


def _get_native():
    global _native_module
    if _native_module is None:
        import _skellytracker_rust
        _native_module = _skellytracker_rust
    return _native_module


# -- Defaults ------------------------------------------------------------------

DEFAULT_MODE = "medium"
DEFAULT_PROVIDER = "cuda"


# -- Rust adapter --------------------------------------------------------------

class RustCompositeGpuTracker(BaseTracker):
    """Adapter wrapping the Rust ``_skellytracker_rust.CompositeGpuTracker``.

    Subclasses ``BaseTracker`` so beartype accepts it anywhere a
    ``BaseTracker`` is expected.
    """

    config: CompositeGPUTrackerConfig
    detector: CompositeGPUDetector
    annotator: CompositeGPUImageAnnotator
    recorder: CompositeGPURecorder | None

    def __init__(
        self,
        mode: str = DEFAULT_MODE,
        provider: str = DEFAULT_PROVIDER,
    ):
        cfg = CompositeGPUTrackerConfig()
        cfg.detector_config.session_config.execution_provider = provider
        detector = CompositeGPUDetector.create(cfg.detector_config)
        annotator = CompositeGPUImageAnnotator.create(cfg.annotator_config)

        super().__init__(
            config=cfg,
            detector=detector,
            annotator=annotator,
            recorder=None,
        )

        native = _get_native()
        self._inner = native.CompositeGpuTracker(mode, provider)

    @classmethod
    def create(cls, config: Any = None):
        mode = DEFAULT_MODE
        provider = DEFAULT_PROVIDER
        if config is not None:
            detector_cfg = getattr(config, "detector_config", None)
            if detector_cfg is not None:
                sess_cfg = getattr(detector_cfg, "session_config", None)
                if sess_cfg is not None:
                    provider = getattr(sess_cfg, "execution_provider", DEFAULT_PROVIDER)
        return cls(mode=mode, provider=provider)

    @property
    def mode(self) -> str:
        return self._inner.mode

    def process_image(
        self, frame_number: int, image: np.ndarray, record_observation: bool = True
    ) -> dict:
        return self._inner.process_image(frame_number, image)

    def annotate_image(self, image: np.ndarray, observation) -> np.ndarray:
        return self._inner.annotate_image(image, observation)

    def __repr__(self) -> str:
        return f"RustCompositeGpuTracker(mode={self._inner.mode})"


# -- Factory -------------------------------------------------------------------

def get_composite_gpu_tracker(
    mode: str = DEFAULT_MODE,
    provider: str = DEFAULT_PROVIDER,
):
    """Return the active CompositeGPU backend based on ``USE_RUST_BACKEND``."""
    if USE_RUST_BACKEND:
        return RustCompositeGpuTracker(mode=mode, provider=provider)
    else:
        cfg = CompositeGPUTrackerConfig()
        cfg.detector_config.session_config.execution_provider = provider
        return CompositeGPUTracker.create(cfg)
