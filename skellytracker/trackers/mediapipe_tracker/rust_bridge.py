"""Hot-swappable Rust backend for MediaPipe tracker.

Phase 1 (reverse PyO3 bridge): Rust ``MediaPipeTracker`` struct holds
``PyObject`` refs to the Python ``MediapipeCompositeDetector`` and
``MediapipeCompositeAnnotator``.  ``detect()`` calls the Python detector
via PyO3, extracts the resulting PointCloud data, and wraps it in a Rust
``MediaPipeObservation``.  ``draw_markers_into()`` delegates to the
Python annotator.

This is a *reverse* bridge compared to BPT/Charuco/RTMPose — Python code
is called FROM Rust rather than Rust being called FROM Python.  The
driver flow is still Python→Rust (webcam demo creates the pyclass), but
internally the Rust tracker calls back into Python for MediaPipe inference.

- ``USE_RUST_BACKEND = True`` selects the Rust PyO3 bridge
- ``USE_RUST_BACKEND = False`` falls back to the native ``MediapipeCompositeTracker``
- ``get_mediapipe_tracker()`` is the single factory function
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
from skellytracker.trackers.mediapipe_tracker.__mediapipe_tracker import (
    MediapipeCompositeTracker,
    MediapipeCompositeRecorder,
)
from skellytracker.trackers.mediapipe_tracker.composite.mediapipe_composite_tracker_config import (
    MediapipeCompositeTrackerConfig,
)
from skellytracker.trackers.mediapipe_tracker.composite.mediapipe_composite_detector import (
    MediapipeCompositeDetector,
)
from skellytracker.trackers.mediapipe_tracker.composite.mediapipe_composite_annotator import (
    MediapipeCompositeAnnotator,
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

# -- Lazy import ---------------------------------------------------------------

_native_module: Any = None


def _get_native():
    global _native_module
    if _native_module is None:
        import _skellytracker_rust

        _native_module = _skellytracker_rust
    return _native_module


# -- Rust adapter --------------------------------------------------------------

class RustMediapipeTracker(BaseTracker):
    """Adapter wrapping the Rust ``_skellytracker_rust.MediaPipeTracker``.

    The Rust struct holds PyObject refs to a Python
    ``MediapipeCompositeDetector`` and ``MediapipeCompositeAnnotator``.
    ``detect()`` calls the Python detector via PyO3, extracts PointCloud
    data, and returns a Rust ``MediaPipeObservation``.  Annotation
    delegates to the Python annotator.

    Subclasses ``BaseTracker`` so beartype accepts it anywhere a
    ``BaseTracker`` is expected.
    """

    config: MediapipeCompositeTrackerConfig
    detector: MediapipeCompositeDetector
    annotator: MediapipeCompositeAnnotator
    recorder: MediapipeCompositeRecorder | None

    def __init__(self, config: MediapipeCompositeTrackerConfig | None = None):
        if config is None:
            config = MediapipeCompositeTrackerConfig()

        python_detector = MediapipeCompositeDetector.create(
            config=config.detector_config
        )
        python_annotator = MediapipeCompositeAnnotator.create(
            config=config.annotator_config
        )

        super().__init__(
            config=config,
            detector=python_detector,
            annotator=python_annotator,
            recorder=None,
        )

        native = _get_native()
        # Pass the Python detector and annotator objects across the bridge.
        # The Rust side stores them as PyObject refs and calls them via PyO3.
        self._inner = native.MediaPipeTracker(
            python_detector,
            python_annotator,
        )

    @classmethod
    def create(cls, config: MediapipeCompositeTrackerConfig | None = None):
        """Match ``MediapipeCompositeTracker.create()`` interface."""
        return cls(config=config)

    def process_image(
        self, frame_number: int, image: np.ndarray, record_observation: bool = True
    ):
        return self._inner.process_image(frame_number, image)

    def annotate_image(self, image: np.ndarray, observation) -> np.ndarray:
        return self._inner.annotate_image(image, observation)

    def __repr__(self) -> str:
        return f"RustMediapipeTracker(config={self.config})"


# -- Factory -------------------------------------------------------------------


def get_mediapipe_tracker(config: MediapipeCompositeTrackerConfig | None = None):
    """Return the active MediaPipe backend based on ``USE_RUST_BACKEND``."""
    if USE_RUST_BACKEND:
        return RustMediapipeTracker.create(config=config)
    else:
        return MediapipeCompositeTracker.create(config=config)
