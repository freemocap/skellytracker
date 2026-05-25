"""Hot-swappable Rust backend for RTMPoseTracker.

Pattern copied from charuco_tracker/rust_bridge.py and brightest_point_tracker/rust_bridge.py.

- ``USE_RUST_BACKEND = True`` selects the Rust PyO3 bridge
- ``USE_RUST_BACKEND = False`` falls back to the original Python ONNX Runtime implementation
- ``get_rtmpose_tracker()`` is the single factory function

OpenCV DLL discovery on Windows:
    The compiled ``_skellytracker_rust.pyd`` links against OpenCV DLLs.
    Before importing, we add the chocolatey OpenCV bin dir to the DLL search path
    via ``os.add_dll_directory()``.
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
from skellytracker.trackers.rtmpose_tracker.rtmpose_detector import (
    RTMPoseDetector,
)
from skellytracker.trackers.rtmpose_tracker.rtmpose_annotator import (
    RTMPoseImageAnnotator,
)
from skellytracker.trackers.rtmpose_tracker.__rtmpose_tracker import (
    RTMPoseTrackerConfig,
)

logger = logging.getLogger(__name__)

# -- Backend selector ------------------------------------------------------------
USE_RUST_BACKEND: bool = True

# -- OpenCV DLL discovery (Windows) ---------------------------------------------

_OPENCV_BIN_DIR = r"C:\tools\opencv\build\x64\vc16\bin"


def _setup_opencv_dlls() -> None:
    if platform.system() != "Windows":
        return
    if not os.path.isdir(_OPENCV_BIN_DIR):
        logger.warning(
            "OpenCV bin dir not found at %s -- Rust tracker import may fail",
            _OPENCV_BIN_DIR,
        )
        return
    try:
        os.add_dll_directory(_OPENCV_BIN_DIR)
    except OSError:
        pass

    current_path = os.environ.get("PATH", "")
    if _OPENCV_BIN_DIR not in current_path:
        os.environ["PATH"] = f"{_OPENCV_BIN_DIR};{current_path}"


_setup_opencv_dlls()

# -- NVIDIA DLL discovery (cuDNN, CUDA runtime) --------------------------------

def _setup_nvidia_dlls() -> None:
    """Ensure NVIDIA CUDA/cuDNN DLLs are discoverable at runtime.

    The ``ort`` crate's ``load-dynamic`` mode loads ``onnxruntime.dll`` at runtime,
    which internally lazy-loads cuDNN sub-DLLs via ``LoadLibrary``. Those calls use
    the process ``PATH``, not the directories registered by ``os.add_dll_directory``.

    Best-effort: if the nvidia pip packages aren't installed, warn but don't crash
    (the user may be running with CPU ORT or have a system CUDA install).
    """
    if platform.system() != "Windows":
        return
    try:
        from importlib.util import find_spec
        spec = find_spec("nvidia")
        if spec is None or not spec.submodule_search_locations:
            logger.warning(
                "nvidia pip namespace package not found — "
                "Rust RTMPose may not find CUDA/cuDNN. "
                "Install with: uv sync --extra rtmpose-gpu"
            )
            return
        from pathlib import Path
        nvidia_root = Path(spec.submodule_search_locations[0])
        bin_dirs = sorted(nvidia_root.glob("*/bin"))
        if not bin_dirs:
            logger.warning(
                f"Found nvidia package at {nvidia_root} but no nvidia/*/bin "
                f"subdirectories — CUDA/cuDNN runtime pip packages not installed."
            )
            return

        # Layer 1: prepend to PATH (cuDNN's internal LoadLibrary calls read this)
        bin_dir_strs = [str(d) for d in bin_dirs]
        os.environ["PATH"] = os.pathsep.join([*bin_dir_strs, os.environ.get("PATH", "")])

        # Layer 2: register with Python's DLL loader
        for bin_dir in bin_dirs:
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass

        # Layer 3: proactively load every cuDNN DLL by full path
        import ctypes
        cudnn_bin = nvidia_root / "cudnn" / "bin"
        if cudnn_bin.is_dir():
            for dll_path in sorted(cudnn_bin.glob("*.dll")):
                try:
                    ctypes.WinDLL(str(dll_path))
                except OSError:
                    pass
        else:
            logger.warning(f"cuDNN bin directory not found at {cudnn_bin}")
    except Exception:
        logger.warning(
            "Failed to set up NVIDIA DLL discovery — "
            "Rust RTMPose may fall back to CPU ONNX Runtime.",
            exc_info=True,
        )


_setup_nvidia_dlls()

# -- Also add ONNX Runtime DLL directory from pip-installed onnxruntime ---------

def _setup_ort_dlls() -> None:
    """Ensure ONNX Runtime DLLs are discoverable (required by ort crate)."""
    if platform.system() != "Windows":
        return
    try:
        import onnxruntime
        ort_dir = os.path.dirname(onnxruntime.__file__)
        ort_bin = os.path.join(ort_dir, "capi")
        if os.path.isdir(ort_bin):
            try:
                os.add_dll_directory(ort_bin)
            except OSError:
                pass
            current_path = os.environ.get("PATH", "")
            if ort_bin not in current_path:
                os.environ["PATH"] = f"{ort_bin};{current_path}"
    except ImportError:
        logger.warning("onnxruntime not installed -- Rust RTMPose may fail")


_setup_ort_dlls()

# -- Pre-load ORT DLL by absolute path before Rust import ---------------------
# The ort crate's `load-dynamic` feature searches PATH for `onnxruntime.dll`.
# If any CPU-only copy exists on PATH, it will be loaded instead of the GPU one.
# Pre-loading the correct DLL by absolute path ensures the right one is used.


def _preload_ort_dll() -> None:
    """Load onnxruntime.dll by absolute path before the Rust module imports it."""
    if platform.system() != "Windows":
        return
    try:
        import onnxruntime
        from pathlib import Path

        ort_dir = Path(onnxruntime.__file__).parent
        ort_dll = ort_dir / "capi" / "onnxruntime.dll"

        if not ort_dll.exists():
            logger.error(
                "ORT DLL not found at expected path: %s — "
                "Rust RTMPose will search PATH and may find wrong DLL.",
                str(ort_dll),
            )
            return

        print(f"[skellytracker-rust bridge] Pre-loading ORT DLL: {ort_dll}")
        import ctypes as _ctypes
        _ctypes.WinDLL(str(ort_dll))
        print(f"[skellytracker-rust bridge] ORT DLL loaded successfully")

        # Also verify CUDA provider DLL is present
        providers_dir = ort_dir / "capi"
        for prov in ["onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll"]:
            prov_path = providers_dir / prov
            if prov_path.exists():
                print(f"[skellytracker-rust bridge]   Found: {prov}")
            else:
                logger.warning("Missing ORT provider DLL: %s", prov)

    except Exception:
        logger.warning(
            "Failed to pre-load ORT DLL — Rust will search PATH instead.",
            exc_info=True,
        )


_preload_ort_dll()

# -- Lazy import ---------------------------------------------------------------

_native_module: Any = None


def _get_native():
    global _native_module
    if _native_module is None:
        import _skellytracker_rust
        _native_module = _skellytracker_rust
    return _native_module


# -- Defaults ------------------------------------------------------------------

DEFAULT_MODE = "balanced"
DEFAULT_PROVIDER = "cuda"


# -- Rust adapter --------------------------------------------------------------

class RustRtmPoseTracker(BaseTracker):
    """Adapter wrapping the Rust ``_skellytracker_rust.RtmPoseTracker``.

    Subclasses ``BaseTracker`` so beartype accepts it anywhere a
    ``BaseTracker`` is expected.
    """

    config: RTMPoseTrackerConfig
    detector: RTMPoseDetector
    annotator: RTMPoseImageAnnotator
    recorder: BaseRecorder | None

    def __init__(
        self,
        mode: str = DEFAULT_MODE,
        provider: str = DEFAULT_PROVIDER,
    ):
        cfg = RTMPoseTrackerConfig()
        cfg.detector_config.mode = mode
        detector = RTMPoseDetector.create(cfg.detector_config)
        annotator = RTMPoseImageAnnotator.create(cfg.annotator_config)

        super().__init__(
            config=cfg,
            detector=detector,
            annotator=annotator,
            recorder=None,
        )

        native = _get_native()
        self._inner = native.RtmPoseTracker(mode, provider)

    @classmethod
    def create(cls, config: Any = None):
        """Match ``RTMPoseTracker.create()`` interface."""
        mode = DEFAULT_MODE
        provider = DEFAULT_PROVIDER
        if config is not None:
            detector_cfg = getattr(config, "detector_config", None)
            if detector_cfg is not None:
                mode = getattr(detector_cfg, "mode", DEFAULT_MODE)
                provider = getattr(detector_cfg, "execution_provider", DEFAULT_PROVIDER)
        return cls(mode=mode, provider=provider)

    @property
    def mode(self) -> str:
        return self._inner.mode

    @property
    def provider(self) -> str:
        return self._inner.provider

    def process_image(
        self, frame_number: int, image: np.ndarray, record_observation: bool = True
    ) -> dict:
        return self._inner.process_image(frame_number, image)

    def annotate_image(self, image: np.ndarray, observation: dict) -> np.ndarray:
        return self._inner.annotate_image(image, observation)

    def __repr__(self) -> str:
        return (
            f"RustRtmPoseTracker(mode={self._inner.mode}, "
            f"provider={self._inner.provider})"
        )


# -- Factory -------------------------------------------------------------------

def get_rtmpose_tracker(mode: str = DEFAULT_MODE, provider: str = DEFAULT_PROVIDER):
    """Return the active RTMPose backend based on ``USE_RUST_BACKEND``."""
    if USE_RUST_BACKEND:
        return RustRtmPoseTracker(mode=mode, provider=provider)
    else:
        from skellytracker.trackers.rtmpose_tracker.__rtmpose_tracker import (
            RTMPoseTracker,
            RTMPoseTrackerConfig,
        )
        cfg = RTMPoseTrackerConfig()
        if hasattr(cfg, "detector_config"):
            cfg.detector_config.mode = mode
        return RTMPoseTracker.create(cfg)
