import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from skellytracker.io.process_videos.video_handler import VideoHandler
from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseTracker


logger = logging.getLogger(__name__)


def process_video(
    *,
    tracker: BaseTracker,
    input_video_path: str | Path,
    output_video_path: str | Path | None = None,
    confidence_threshold: float | None = None,
    fill_with_nans: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Process one video with an already-created tracker.

    Returns
    -------
    np.ndarray
        Pixel coordinates with shape:

        frames × tracked_points × 2
    """

    input_video_path = Path(input_video_path)

    if not input_video_path.is_file():
        raise FileNotFoundError(
            f"Input video does not exist: {input_video_path}"
        )

    capture = cv2.VideoCapture(str(input_video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"OpenCV could not open video: {input_video_path}"
        )

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))

    if fps <= 0:
        fps = 30.0

    video_handler: VideoHandler | None = None

    if output_video_path is not None:
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        video_handler = VideoHandler(
            output_path=output_video_path,
            frame_size=(frame_width, frame_height),
            fps=fps,
        )

    progress = None

    if show_progress:
        progress = tqdm(
            total=frame_count if frame_count > 0 else None,
            desc=f"Processing {input_video_path.name}",
            unit="frames",
            dynamic_ncols=True,
        )

    frame_arrays: list[np.ndarray] = []
    expected_point_shape: tuple[int, int] | None = None
    frame_number = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            observation = tracker.process_image(
                frame_number=frame_number,
                image=frame,
                record_observation=False,
            )

            points_2d = np.asarray(
                observation.to_2d_array(
                    confidence_threshold=confidence_threshold,
                    fill_with_nans=fill_with_nans,
                ),
                dtype=np.float32,
            )

            if points_2d.ndim != 2 or points_2d.shape[-1] != 2:
                raise ValueError(
                    f"{observation.tracker_type} returned unexpected "
                    f"2D data shape {points_2d.shape}; expected "
                    f"(tracked_points, 2)."
                )

            if expected_point_shape is None:
                expected_point_shape = points_2d.shape
            elif points_2d.shape != expected_point_shape:
                raise ValueError(
                    f"Tracked-point shape changed at frame {frame_number}. "
                    f"Expected {expected_point_shape}, received "
                    f"{points_2d.shape}."
                )

            frame_arrays.append(points_2d)

            if video_handler is not None:
                if tracker.annotator is None:
                    raise RuntimeError(
                        f"{tracker.__class__.__name__} does not have "
                        f"an annotator."
                    )

                annotated_frame = tracker.annotate_image(
                    image=frame,
                    observation=observation,
                )

                video_handler.add_frame(annotated_frame)

            frame_number += 1

            if progress is not None:
                progress.update(1)

    finally:
        capture.release()

        if video_handler is not None:
            video_handler.close()

        if progress is not None:
            progress.close()

    if not frame_arrays:
        raise RuntimeError(
            f"No frames were processed from: {input_video_path}"
        )

    return np.stack(frame_arrays, axis=0)