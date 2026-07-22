from __future__ import annotations

import multiprocessing as mp
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from skellytracker.io.process_videos.process_single_video import (
    process_video,
)
from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseTracker,
)


TrackerFactory = Callable[[], BaseTracker]


@dataclass(frozen=True)
class _ProcessVideoArgs:
    """
    Pickleable arguments passed to one video-processing worker.
    """

    camera_index: int
    video_path: Path
    tracker_factory: TrackerFactory
    output_video_path: Path | None
    confidence_threshold: float | None
    fill_with_nans: bool
    progress_queue: object | None


def _process_one_video(
    args: _ProcessVideoArgs,
) -> tuple[int, np.ndarray]:
    """
    Process one camera video inside a child process.

    This function must remain at module scope for Windows multiprocessing.
    """

    tracker = args.tracker_factory()

    try:
        camera_array = process_video(
            tracker=tracker,
            input_video_path=args.video_path,
            output_video_path=args.output_video_path,
            confidence_threshold=args.confidence_threshold,
            fill_with_nans=args.fill_with_nans,
            # The parent process displays the shared progress bar.
            show_progress=False,
            progress_queue=args.progress_queue,
        )

        return args.camera_index, camera_array

    finally:
        _close_tracker_resources(tracker)


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
    num_workers: int | None = None,
) -> np.ndarray:
    """
    Process an explicitly ordered collection of synchronized videos.

    Each camera video is processed in a separate worker process. Frames
    within an individual camera remain sequential so that the tracker's
    temporal state is preserved.

    Returns
    -------
    np.ndarray
        Pixel coordinates with shape:

        cameras × frames × tracked_points × 2
    """

    ordered_video_paths = tuple(
        Path(path)
        for path in video_paths
    )

    if not ordered_video_paths:
        raise ValueError("No video paths were provided.")

    for video_path in ordered_video_paths:
        if not video_path.exists():
            raise FileNotFoundError(
                f"Video does not exist: {video_path}"
            )

    annotated_directory: Path | None = None

    if annotated_video_directory is not None:
        annotated_directory = Path(annotated_video_directory)
        annotated_directory.mkdir(parents=True, exist_ok=True)

    camera_count = len(ordered_video_paths)

    if num_workers is None:
        num_workers = min(
            camera_count,
            max(1, mp.cpu_count() - 1),
        )

    if num_workers < 1:
        raise ValueError("num_workers must be at least 1.")

    num_workers = min(num_workers, camera_count)

    expected_frame_counts = [
        _get_video_frame_count(video_path)
        for video_path in ordered_video_paths
    ]

    expected_total_frames = sum(expected_frame_counts)

    context = mp.get_context("spawn")

    manager = context.Manager() if show_progress else None
    progress_queue = (
        manager.Queue()
        if manager is not None
        else None
    )

    progress_bar = (
        tqdm(
            total=expected_total_frames or None,
            desc=f"{tracker_name} — all cameras",
            unit="frame",
            dynamic_ncols=True,
        )
        if show_progress
        else None
    )

    progress_thread: threading.Thread | None = None

    def consume_progress() -> None:
        if progress_queue is None or progress_bar is None:
            return

        while True:
            message = progress_queue.get()

            # None is the stop sentinel. Workers only send integers.
            if message is None:
                break

            progress_bar.update(int(message))

    if show_progress:
        progress_thread = threading.Thread(
            target=consume_progress,
            daemon=True,
        )
        progress_thread.start()

    worker_arguments: list[_ProcessVideoArgs] = []

    for camera_index, video_path in enumerate(
        ordered_video_paths
    ):
        output_video_path: Path | None = None

        if annotated_directory is not None:
            output_video_path = (
                annotated_directory
                / (
                    f"{video_path.stem}_"
                    f"{tracker_name}_annotated.mp4"
                )
            )

        worker_arguments.append(
            _ProcessVideoArgs(
                camera_index=camera_index,
                video_path=video_path,
                tracker_factory=tracker_factory,
                output_video_path=output_video_path,
                confidence_threshold=confidence_threshold,
                fill_with_nans=fill_with_nans,
                progress_queue=progress_queue,
            )
        )

    indexed_camera_arrays: list[
        tuple[int, np.ndarray]
    ] = []

    try:
        with context.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(
                _process_one_video,
                worker_arguments,
            ):
                indexed_camera_arrays.append(result)

    finally:
        if progress_queue is not None:
            progress_queue.put(None)

        if progress_thread is not None:
            progress_thread.join(timeout=5)

        if progress_bar is not None:
            progress_bar.close()

        if manager is not None:
            manager.shutdown()

    if len(indexed_camera_arrays) != camera_count:
        raise RuntimeError(
            f"Expected results from {camera_count} cameras, but "
            f"received {len(indexed_camera_arrays)}."
        )

    # imap_unordered returns videos in completion order. Restore the
    # caller's explicit camera ordering before stacking.
    indexed_camera_arrays.sort(
        key=lambda result: result[0]
    )

    camera_arrays = [
        camera_array
        for _, camera_array in indexed_camera_arrays
    ]

    point_shapes = {
        camera_array.shape[1:]
        for camera_array in camera_arrays
    }

    if len(point_shapes) != 1:
        raise ValueError(
            "Trackers produced inconsistent point-array shapes across "
            f"cameras: {sorted(point_shapes)}"
        )

    actual_frame_counts = [
        camera_array.shape[0]
        for camera_array in camera_arrays
    ]

    if len(set(actual_frame_counts)) != 1:
        if enforce_equal_frame_counts:
            raise ValueError(
                "Synchronized videos produced different frame counts: "
                f"{actual_frame_counts}"
            )

        minimum_frame_count = min(actual_frame_counts)

        camera_arrays = [
            camera_array[:minimum_frame_count]
            for camera_array in camera_arrays
        ]

    return np.stack(camera_arrays, axis=0)


def _get_video_frame_count(video_path: Path) -> int:
    """
    Read a video's reported frame count for the shared progress bar.
    """

    capture = cv2.VideoCapture(str(video_path))

    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        return int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

    finally:
        capture.release()


def _close_tracker_resources(
    tracker: BaseTracker,
) -> None:
    """
    Close an underlying model resource when it exposes close().

    MediaPipe Holistic exposes close(); the other trackers may not.
    """

    detector_resource = getattr(
        tracker.detector,
        "detector",
        None,
    )

    close = getattr(
        detector_resource,
        "close",
        None,
    )

    if callable(close):
        close()