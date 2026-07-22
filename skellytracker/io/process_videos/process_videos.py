from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from skellytracker.io.process_videos.process_single_video import (
    process_video,
)
from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseTracker,
)


TrackerFactory = Callable[[], BaseTracker]


def process_videos(
    *,
    video_paths: Sequence[str | Path],
    tracker_factory: TrackerFactory,
    tracker_name: str,
    annotated_video_directory: str | Path | None = None,
    confidence_threshold: float | None = None,
    fill_with_nans: bool = True,
    enforce_equal_frame_counts: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Process an explicitly ordered collection of synchronized videos.

    Returns
    -------
    np.ndarray
        Pixel coordinates with shape:

        cameras × frames × tracked_points × 2
    """

    ordered_video_paths = tuple(Path(path) for path in video_paths)

    if not ordered_video_paths:
        raise ValueError("No video paths were provided.")

    annotated_directory: Path | None = None

    if annotated_video_directory is not None:
        annotated_directory = Path(annotated_video_directory)
        annotated_directory.mkdir(parents=True, exist_ok=True)

    camera_arrays: list[np.ndarray] = []

    for camera_index, video_path in enumerate(ordered_video_paths):
        # Each camera needs independent temporal tracker state.
        tracker = tracker_factory()

        output_video_path = None

        if annotated_directory is not None:
            output_video_path = (
                annotated_directory
                / f"{video_path.stem}_{tracker_name}_annotated.mp4"
            )

        camera_array = process_video(
            tracker=tracker,
            input_video_path=video_path,
            output_video_path=output_video_path,
            confidence_threshold=confidence_threshold,
            fill_with_nans=fill_with_nans,
            show_progress=show_progress,
        )

        camera_arrays.append(camera_array)

        _close_tracker_resources(tracker)

    point_shapes = {
        camera_array.shape[1:]
        for camera_array in camera_arrays
    }

    if len(point_shapes) != 1:
        raise ValueError(
            "Trackers produced inconsistent point-array shapes across "
            f"cameras: {sorted(point_shapes)}"
        )

    frame_counts = [
        camera_array.shape[0]
        for camera_array in camera_arrays
    ]

    if len(set(frame_counts)) != 1:
        if enforce_equal_frame_counts:
            raise ValueError(
                "Synchronized videos produced different frame counts: "
                f"{frame_counts}"
            )

        minimum_frame_count = min(frame_counts)

        camera_arrays = [
            camera_array[:minimum_frame_count]
            for camera_array in camera_arrays
        ]

    return np.stack(camera_arrays, axis=0)


def _close_tracker_resources(tracker: BaseTracker) -> None:
    """
    Close an underlying model resource when it exposes a close method.

    MediaPipe Holistic exposes close(); other trackers may not.
    """

    detector_resource = getattr(
        tracker.detector,
        "detector",
        None,
    )

    close = getattr(detector_resource, "close", None)

    if callable(close):
        close()