"""Hot-swappable Rust backend for CharucoTracker.

Pattern copied from brightest_point_tracker/rust_bridge.py:

- ``USE_RUST_BACKEND = True`` selects the Rust PyO3 bridge
- ``USE_RUST_BACKEND = False`` falls back to the original Python OpenCV implementation
- ``get_charuco_tracker()`` is the single factory function — callers don't
  need to know which backend they're getting

OpenCV DLL discovery on Windows:
    The compiled ``_skellytracker_rust.pyd`` links against OpenCV DLLs.
    Before importing, we add the chocolatey OpenCV bin dir to the DLL search path
    via ``os.add_dll_directory()``.
"""

import logging
import os
import platform
from typing import Any

import cv2
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseTracker,
    BaseTrackerConfig,
    BaseDetector,
    BaseImageAnnotator,
    BaseRecorder,
)
from skellytracker.trackers.charuco_tracker.charuco_tracker_config import (
    CharucoTrackerConfig,
    CharucoDetectorConfig,
)
from skellytracker.trackers.charuco_tracker.charuco_detector import CharucoDetector
from skellytracker.trackers.charuco_tracker.charuco_annotator import (
    CharucoImageAnnotator,
    CharucoAnnotatorConfig,
)

logger = logging.getLogger(__name__)

# ── Backend selector ────────────────────────────────────────────────────────
USE_RUST_BACKEND: bool = True

# ── OpenCV DLL discovery (Windows) ───────────────────────────────────────────

_OPENCV_BIN_DIR = r"C:\tools\opencv\build\x64\vc16\bin"


def _setup_opencv_dlls() -> None:
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
        pass

    current_path = os.environ.get("PATH", "")
    if _OPENCV_BIN_DIR not in current_path:
        os.environ["PATH"] = f"{_OPENCV_BIN_DIR};{current_path}"


_setup_opencv_dlls()

# ── Lazy import ──────────────────────────────────────────────────────────────

_native_module: Any = None


def _get_native():
    global _native_module
    if _native_module is None:
        import _skellytracker_rust
        _native_module = _skellytracker_rust
    return _native_module


# ── Board defaults (matching CharucoBoardDefinition.create_letter_size_5x3) ──

DEFAULT_SQUARES_X = 5
DEFAULT_SQUARES_Y = 3
DEFAULT_SQUARE_LENGTH_MM = 54.0
DEFAULT_MARKER_LENGTH_RATIO = 0.8
DEFAULT_DICTIONARY_ENUM = cv2.aruco.DICT_4X4_250


# ── Rust adapter ─────────────────────────────────────────────────────────────

class RustCharucoTracker(BaseTracker):
    """Adapter wrapping the Rust ``_skellytracker_rust.CharucoTracker``.

    Subclasses ``BaseTracker`` so beartype accepts it anywhere a
    ``BaseTracker`` is expected.  The ``config`` / ``detector`` / ``annotator``
    fields are populated with lightweight Python stubs — ``process_image`` and
    ``annotate_image`` are overridden to delegate directly to the Rust engine.
    """

    config: CharucoTrackerConfig
    detector: CharucoDetector
    annotator: CharucoImageAnnotator
    recorder: BaseRecorder | None

    def __init__(
        self,
        squares_x: int = DEFAULT_SQUARES_X,
        squares_y: int = DEFAULT_SQUARES_Y,
        square_length_mm: float = DEFAULT_SQUARE_LENGTH_MM,
        marker_length_ratio: float = DEFAULT_MARKER_LENGTH_RATIO,
        dictionary_enum: int = DEFAULT_DICTIONARY_ENUM,
    ):
        cfg = CharucoTrackerConfig()
        cfg.detector_config.board.squares_x = squares_x
        cfg.detector_config.board.squares_y = squares_y
        cfg.detector_config.board.square_length_mm = square_length_mm
        cfg.detector_config.board.marker_length_ratio = marker_length_ratio
        cfg.detector_config.board.aruco_dictionary_enum = dictionary_enum
        detector = CharucoDetector.create(cfg.detector_config)
        annotator = CharucoImageAnnotator.create(cfg.annotator_config)

        super().__init__(
            config=cfg,
            detector=detector,
            annotator=annotator,
            recorder=None,
        )

        native = _get_native()
        self._inner = native.CharucoTracker(
            squares_x, squares_y, square_length_mm, marker_length_ratio, dictionary_enum
        )

    @classmethod
    def create(cls, config: CharucoTrackerConfig | None = None):
        """Match ``CharucoTracker.create()`` interface."""
        kwargs = {}
        if config is not None:
            detector_cfg = getattr(config, "detector_config", None)
            if detector_cfg is not None:
                board = getattr(detector_cfg, "board", None)
                if board is not None:
                    kwargs["squares_x"] = getattr(board, "squares_x", DEFAULT_SQUARES_X)
                    kwargs["squares_y"] = getattr(board, "squares_y", DEFAULT_SQUARES_Y)
                    kwargs["square_length_mm"] = getattr(board, "square_length_mm", DEFAULT_SQUARE_LENGTH_MM)
                    kwargs["marker_length_ratio"] = getattr(board, "marker_length_ratio", DEFAULT_MARKER_LENGTH_RATIO)
                    kwargs["dictionary_enum"] = getattr(board, "aruco_dictionary_enum", DEFAULT_DICTIONARY_ENUM)
        return cls(**kwargs)

    @property
    def squares_x(self) -> int:
        return self._inner.squares_x

    @property
    def squares_y(self) -> int:
        return self._inner.squares_y

    @property
    def all_charuco_ids(self) -> list[int]:
        return list(self._inner.all_charuco_ids)

    @property
    def all_aruco_ids(self) -> list[int]:
        return list(self._inner.all_aruco_ids)

    def process_image(
        self, frame_number: int, image: np.ndarray, record_observation: bool = True
    ) -> dict:
        return self._inner.process_image(frame_number, image)

    def annotate_image(self, image: np.ndarray, observation: dict) -> np.ndarray:
        return self._inner.annotate_image(image, observation)

    def __repr__(self) -> str:
        return (
            f"RustCharucoTracker("
            f"squares_x={self._inner.squares_x}, "
            f"squares_y={self._inner.squares_y})"
        )


# ── Factory ──────────────────────────────────────────────────────────────────

def get_charuco_tracker(
    squares_x: int = DEFAULT_SQUARES_X,
    squares_y: int = DEFAULT_SQUARES_Y,
    square_length_mm: float = DEFAULT_SQUARE_LENGTH_MM,
    marker_length_ratio: float = DEFAULT_MARKER_LENGTH_RATIO,
    dictionary_enum: int = DEFAULT_DICTIONARY_ENUM,
):
    """Return the active Charuco backend based on ``USE_RUST_BACKEND``."""
    if USE_RUST_BACKEND:
        return RustCharucoTracker(
            squares_x=squares_x,
            squares_y=squares_y,
            square_length_mm=square_length_mm,
            marker_length_ratio=marker_length_ratio,
            dictionary_enum=dictionary_enum,
        )
    else:
        from skellytracker.trackers.charuco_tracker.__charuco_tracker import (
            CharucoTracker,
        )
        from skellytracker.trackers.charuco_tracker.charuco_board_definition import (
            CharucoBoardDefinition,
        )

        board = CharucoBoardDefinition(
            squares_x=squares_x,
            squares_y=squares_y,
            square_length_mm=square_length_mm,
            marker_length_ratio=marker_length_ratio,
            aruco_dictionary_enum=dictionary_enum,
        )
        config = CharucoTrackerConfig()
        config.detector_config.board = board
        return CharucoTracker.create(config)
