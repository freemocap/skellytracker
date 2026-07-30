from dataclasses import dataclass, field

from skellytracker.core.io.processing_timer import ProcessingTimer
from skellytracker.core.tracker.task_events import TrackerTaskEventCollector


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
    event_collector: Optional task-event sink. When set, DetectionStage.run_batch
                     emits per-stage TrackerTaskEvent records for pipeline metrics.
    """

    frame_number: int = 0
    timestamp_ms: int | None = None
    timings: ProcessingTimer | None = field(default=None, repr=False)
    event_collector: TrackerTaskEventCollector | None = field(default=None, repr=False)
