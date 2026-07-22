from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from skellytracker.io.process_videos.video_handler import VideoHandler
from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseTracker


def process_video(
    *,
    tracker: BaseTracker,
    input_video_path: str | Path,
    output_video_path: str | Path | None = None,
    confidence_threshold: float | None = None,
    fill_with_nans: bool = True,
    show_progress: bool = True,
    progress_queue: object | None = None,
) -> np.ndarray:
    """
    Process one video sequentially with one tracker instance.

    Parameters
    ----------
    tracker
        Tracker instance used to process every frame in the video.
    input_video_path
        Video that will be processed.
    output_video_path
        Optional path for an annotated video.
    confidence_threshold
        Optional confidence threshold passed to each observation.
    fill_with_nans
        Whether low-confidence points should be replaced with NaNs.
    show_progress
        Show a local progress bar when not being run through the
        multi-video multiprocessing runner.
    progress_queue
        Optional multiprocessing queue used to report completed frames
        to a parent process.

    Returns
    -------
    np.ndarray
        Tracked pixel coordinates with shape:

        frames × tracked_points × 2
    """

    input_video_path = Path(input_video_path)

    if not input_video_path.exists():
        raise FileNotFoundError(
            f"Input video does not exist: {input_video_path}"
        )

    output_path: Path | None = None

    if output_video_path is not None:
        output_path = Path(output_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_video_path))

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open video: {input_video_path}"
        )

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        fps = 30.0

    video_handler: VideoHandler | None = None

    if output_path is not None:
        video_handler = VideoHandler(
            output_path=output_path,
            frame_size=(width, height),
            fps=fps,
        )

    # A worker process reports progress to the shared parent progress bar.
    # A direct call to process_video gets its own local progress bar.
    use_local_progress = show_progress and progress_queue is None

    progress_bar = (
        tqdm(
            total=expected_frame_count or None,
            desc=f"Processing {input_video_path.name}",
            unit="frame",
            dynamic_ncols=True,
        )
        if use_local_progress
        else None
    )

    frame_arrays: list[np.ndarray] = []
    frame_number = 0

    try:
        while True:
            success, frame = capture.read()

            if not success or frame is None:
                break

            observation = tracker.process_image(
                frame_number=frame_number,
                image=frame,
                # Storing full observations can consume a large amount of
                # memory, especially with MediaPipe segmentation masks.
                record_observation=False,
            )

            frame_array = np.asarray(
                observation.to_2d_array(
                    confidence_threshold=confidence_threshold,
                    fill_with_nans=fill_with_nans,
                )
            )

            if frame_array.ndim != 2 or frame_array.shape[-1] != 2:
                raise ValueError(
                    f"Tracker produced an invalid array for frame "
                    f"{frame_number} of {input_video_path.name}. "
                    f"Expected (tracked_points, 2), received "
                    f"{frame_array.shape}."
                )

            frame_arrays.append(frame_array)

            if video_handler is not None:
                annotated_frame = tracker.annotate_image(
                    image=frame,
                    observation=observation,
                )
                video_handler.add_frame(annotated_frame)

            if progress_queue is not None:
                progress_queue.put(1)

            if progress_bar is not None:
                progress_bar.update(1)

            frame_number += 1

    finally:
        capture.release()

        if video_handler is not None:
            video_handler.close()

        if progress_bar is not None:
            progress_bar.close()

    if not frame_arrays:
        raise ValueError(
            f"No frames were successfully processed from "
            f"{input_video_path}"
        )

    try:
        return np.stack(frame_arrays, axis=0)

    except ValueError as error:
        frame_shapes = sorted(
            {frame_array.shape for frame_array in frame_arrays}
        )

        raise ValueError(
            "Tracker produced inconsistent point-array shapes within "
            f"{input_video_path.name}: {frame_shapes}"
        ) from error