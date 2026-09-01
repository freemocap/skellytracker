from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skellytracker.core.io.processing_timer import ProcessingTimer


@dataclass
class DetectionContext:
    """Per-frame metadata that detectors may use beyond the image itself.

    Passed from Tracker.process_image() down through DetectionStage to every
    detector. Fields are optional so detectors that don't need them can ignore
    the context entirely.

    frame_number: Index of the current frame (0-based).
    timestamp_ms: Monotonically increasing wall-clock time in milliseconds.
                  Required by detectors running in VIDEO mode (e.g. MediaPipe).
                  When None, detectors that need a timestamp should derive one
                  from time.monotonic().
    timings:      Optional profiler. When set, DetectionStage.run_batch records
                  per-section wall-clock times into it.
    """

    frame_number: int = 0
    timestamp_ms: int | None = None
    timings: ProcessingTimer | None = field(default=None, repr=False)
    parent_keypoints: Any | None = None
