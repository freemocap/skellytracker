"""Hot-swappable Rust backend for BrightestPointTracker.

Pattern copied from skellycam's ``camera_group_manager.py``:

- ``USE_RUST_BACKEND = True`` selects the Rust PyO3 bridge
- ``USE_RUST_BACKEND = False`` falls back to the original Python OpenCV implementation
- ``get_brightest_point_tracker()`` is the single factory function — callers don't
  need to know which backend they're getting

OpenCV DLL discovery on Windows:
    The compiled ``_skellytracker_rust.pyd`` links against OpenCV DLLs.
    Before importing, we add the chocolatey OpenCV bin dir to the DLL search path
    via ``os.add_dll_directory()``.  This replaces the old approach of copying
    DLLs into a ``python/`` package directory (which no longer exists).
"""

import logging
import os
import platform
import sys
from typing import Any

import numpy as np

from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseTracker,
    BaseTrackerConfig,
    BaseDetector,
    BaseImageAnnotator,
    BaseRecorder,
)
from skellytracker.trackers.brightest_point_tracker.brightest_point_detector import (
    BrightestPointDetector,
    BrightestPointDetectorConfig,
)
from skellytracker.trackers.brightest_point_tracker.brightest_point_annotator import (
    BrightestPointImageAnnotator,
    BrightestPointAnnotatorConfig,
)
from skellytracker.trackers.brightest_point_tracker.__brightest_point_tracker import (
    BrightestPointTrackerConfig,
)

logger = logging.getLogger(__name__)

# ── Backend selector ────────────────────────────────────────────────────────
# True  → Rust OpenCV engine (PyO3)
# False → Python OpenCV engine (original)
USE_RUST_BACKEND: bool = True

# ── OpenCV DLL discovery (Windows) ───────────────────────────────────────────

_OPENCV_BIN_DIR = r"C:\tools\opencv\build\x64\vc16\bin"


def _setup_opencv_dlls() -> None:
    """Add the OpenCV bin directory to the Windows DLL search path.

    Must be called BEFORE ``import _skellytracker_rust`` so the OS loader
    finds ``opencv_world4130.dll`` and friends when loading the ``.pyd``.
    """
    if platform.system() != "Windows":
        return
    if not os.path.isdir(_OPENCV_BIN_DIR):
        logger.warning(
            "OpenCV bin dir not found at %s — Rust tracker import may fail",
            _OPENCV_BIN_DIR,
        )
        return
    try:
        os.add_dll_directory(_OPENCV_BIN_DIR)
    except OSError:
        # Already added (e.g. called a second time or from another module).
        pass

    # Also put the bin dir on PATH so any transitive DLL loads work.
    current_path = os.environ.get("PATH", "")
    if _OPENCV_BIN_DIR not in current_path:
        os.environ["PATH"] = f"{_OPENCV_BIN_DIR};{current_path}"


_setup_opencv_dlls()

# ── Lazy import of the native module ────────────────────────────────────────

_native_module: Any = None  # holds the _skellytracker_rust module after first import


def _get_native():
    global _native_module
    if _native_module is None:
        import _skellytracker_rust
        _native_module = _skellytracker_rust
    return _native_module


# ── Rust adapter ────────────────────────────────────────────────────────────

class RustBrightestPointTracker(BaseTracker):
    """Adapter wrapping the Rust ``_skellytracker_rust.BrightestPointTracker``.

    Subclasses ``BaseTracker`` so beartype accepts it anywhere a
    ``BaseTracker`` is expected (WebcamDemoViewer, etc).  The
    ``config`` / ``detector`` / ``annotator`` / ``recorder`` fields
    are populated with lightweight Python stubs that are never used
    for detection — ``process_image`` and ``annotate_image`` are
    overridden to delegate directly to the Rust native module.
    """

    config: BrightestPointTrackerConfig
    detector: BrightestPointDetector
    annotator: BrightestPointImageAnnotator
    recorder: BaseRecorder | None

    def __init__(self, num_points: int = 1, luminance_threshold: int = 200):
        # Build minimal stubs to satisfy the BaseTracker dataclass contract.
        # These are never called for detection/annotation — those methods are
        # overridden below and delegate to the Rust inner tracker.
        cfg = BrightestPointTrackerConfig()
        cfg.detector_config.num_tracked_points = num_points
        cfg.detector_config.luminance_threshold = luminance_threshold
        detector = BrightestPointDetector.create(cfg.detector_config)
        annotator = BrightestPointImageAnnotator.create(cfg.annotator_config)

        super().__init__(
            config=cfg,
            detector=detector,
            annotator=annotator,
            recorder=None,
        )

        native = _get_native()
        self._inner = native.BrightestPointTracker(num_points, luminance_threshold)

    @classmethod
    def create(cls, config: BrightestPointTrackerConfig | None = None):
        """Match ``BrightestPointTracker.create()`` interface."""
        num_points = 1
        luminance_threshold = 200
        if config is not None:
            detector_cfg = getattr(config, "detector_config", None)
            if detector_cfg is not None:
                num_points = getattr(detector_cfg, "num_tracked_points", num_points)
                luminance_threshold = getattr(detector_cfg, "luminance_threshold", luminance_threshold)
        return cls(num_points=num_points, luminance_threshold=luminance_threshold)

    @property
    def num_points(self) -> int:
        return self._inner.num_points

    @property
    def luminance_threshold(self) -> int:
        return self._inner.luminance_threshold

    def process_image(self, frame_number: int, image: np.ndarray, record_observation: bool = True) -> dict:
        """Run detection via Rust. Returns a dict with xy, visibility, etc."""
        return self._inner.process_image(frame_number, image)

    def annotate_image(self, image: np.ndarray, observation: dict) -> np.ndarray:
        """Draw cross markers from a previous ``process_image`` result (Rust)."""
        return self._inner.annotate_image(image, observation)

    def __repr__(self) -> str:
        return (
            f"RustBrightestPointTracker("
            f"num_points={self._inner.num_points}, "
            f"luminance_threshold={self._inner.luminance_threshold})"
        )


# ── Factory ──────────────────────────────────────────────────────────────────

def get_brightest_point_tracker(
    num_points: int = 1,
    luminance_threshold: int = 200,
):
    """Return the active backend based on ``USE_RUST_BACKEND``.

    Returns
    -------
    RustBrightestPointTracker  if USE_RUST_BACKEND is True
    BrightestPointTracker      if USE_RUST_BACKEND is False (original Python)
    """
    if USE_RUST_BACKEND:
        return RustBrightestPointTracker(
            num_points=num_points,
            luminance_threshold=luminance_threshold,
        )
    else:
        # Deferred import — the Python backend pulls in heavier deps (pydantic, etc).
        from skellytracker.trackers.brightest_point_tracker.brightest_point_tracker import (
            BrightestPointTracker,
        )
        return BrightestPointTracker(
            num_points=num_points,
            luminance_threshold=luminance_threshold,
        )
