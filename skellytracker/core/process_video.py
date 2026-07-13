"""Batch video processing with data saving.

Runs a Tracker frame-by-frame on one or more video files and writes the
resulting keypoint arrays to disk.  The primary output is a
(num_frames, num_points, 3) .npy file per video, matching the format
expected by freemocap triangulation.

Typical usage::

    store = process_video(tracker, annotator, Path("cam0.mp4"), output_dir=Path("output/"))
    # saves output/cam0.npy

    process_folder(tracker, annotator, Path("synchronized_videos/"), output_dir=Path("output/"))
    # saves output/cam0.npy, output/cam1.npy, ...
"""
from __future__ import annotations

import concurrent.futures
import logging
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from tqdm import tqdm

from skellytracker.core.annotation.annotator import Annotator
from skellytracker.core.data_store import DataStore
from skellytracker.core.processing_timer import ProcessingTimer
from skellytracker.core.tracker.tracker import Tracker
from skellytracker.core.tracker.tracker_state import TrackerState

logger = logging.getLogger(__name__)

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def process_video(
    tracker: Tracker,
    annotator: Annotator | None,
    input_path: Path,
    output_dir: Path,
    annotated_video_path: Path | None = None,
    fmt: Literal["npy", "json"] = "npy",
    show_progress: bool = True,
) -> DataStore:
    """Run *tracker* on every frame of *input_path* and save results.

    Args:
        tracker: Configured Tracker instance (already initialised; not closed here).
        annotator: If provided, draws landmarks on each frame. Required when
            *annotated_video_path* is set; ignored otherwise.
        input_path: Path to the input video file.
        output_dir: Directory where the keypoint array is written.
            The file is named ``<input_stem>.<fmt>``.
        annotated_video_path: If set, write an annotated video here.
            *annotator* must be provided.
        fmt: ``"npy"`` (default) or ``"json"``.
        show_progress: Show a tqdm progress bar.

    Returns:
        The populated DataStore (also saved to disk).
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer: cv2.VideoWriter | None = None
    if annotated_video_path is not None:
        if annotator is None:
            raise ValueError("annotator must be provided when annotated_video_path is set")
        annotated_video_path = Path(annotated_video_path)
        annotated_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (width, height))

    store = DataStore()
    state = TrackerState()
    frame_number = 0

    iterator = tqdm(
        range(n_frames),
        desc=input_path.name,
        unit="frames",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    try:
        for _ in iterator:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            timestamp_ms = int(frame_number / fps * 1000)
            observation, state = tracker.process_image(
                frame, frame_number, state, timestamp_ms
            )
            store.add(observation)

            if video_writer is not None:
                annotated = annotator.annotate(frame, observation)
                video_writer.write(annotated)

            frame_number += 1
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()

    output_path = output_dir / f"{input_path.stem}.{fmt}"
    store.save(output_path, fmt=fmt)
    logger.info(f"Saved {frame_number} frames → {output_path}")
    return store


def process_folder(
    tracker: Tracker,
    annotator: Annotator | None,
    video_dir: Path,
    output_dir: Path,
    annotated_video_dir: Path | None = None,
    fmt: Literal["npy", "json"] = "npy",
    show_progress: bool = True,
    profile: bool = False,
) -> dict[str, DataStore]:
    """Process all videos in *video_dir* with batched inference.

    Opens all N camera videos simultaneously and calls ``tracker.process_batch()``
    once per frame, so ONNX-backed detectors make a single GPU call per model per
    frame rather than N sequential calls.  Suitable for synchronised multi-camera
    freemocap recordings where all videos cover the same time range.

    For best GPU utilisation, create the ``OnnxSession`` with
    ``batch_size`` equal to the number of camera videos in *video_dir*.

    Args:
        tracker: Configured Tracker instance (already initialised; not closed here).
        annotator: If provided, draws landmarks on each frame. Required when
            *annotated_video_dir* is set; ignored otherwise.
        video_dir: Directory containing the synchronised video files.
        output_dir: Directory where one keypoint array per camera is written.
            Files are named ``<video_stem>.<fmt>``.
        annotated_video_dir: If set, write one annotated video per camera here.
            *annotator* must be provided.
        fmt: ``"npy"`` (default) or ``"json"``.
        show_progress: Show a tqdm progress bar over frames.

    Returns:
        Mapping of video stem → DataStore for each processed video.
    """
    video_dir = Path(video_dir)
    video_paths = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES
    )
    if not video_paths:
        raise FileNotFoundError(f"No video files found in {video_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reset per-camera detector instances so repeated calls to this function
    # don't inherit timestamp state from a previous recording.
    tracker.reset_temporal_state()

    captures = {p.stem: cv2.VideoCapture(str(p)) for p in video_paths}
    cam_ids = list(captures.keys())

    for cam_id, cap in captures.items():
        if not cap.isOpened():
            for c in captures.values():
                c.release()
            raise RuntimeError(f"Could not open video for camera {cam_id!r}")

    frame_counts = {cam_id: int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cam_id, cap in captures.items()}
    unique_counts = set(frame_counts.values())
    if len(unique_counts) > 1:
        logger.warning("Camera videos have different frame counts: %s — processing to shortest", frame_counts)
    min_frames = min(frame_counts.values())

    first_cap = next(iter(captures.values()))
    fps = first_cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writers: dict[str, cv2.VideoWriter] = {}
    if annotated_video_dir is not None:
        if annotator is None:
            raise ValueError("annotator must be provided when annotated_video_dir is set")
        Path(annotated_video_dir).mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for cam_id in cam_ids:
            writers[cam_id] = cv2.VideoWriter(
                str(Path(annotated_video_dir) / f"{cam_id}_annotated.mp4"),
                fourcc, fps, (width, height),
            )

    stores = {cam_id: DataStore() for cam_id in cam_ids}
    states: dict[str, TrackerState] = {cam_id: TrackerState() for cam_id in cam_ids}
    frame_number = 0
    timer = ProcessingTimer() if profile else None

    iterator = tqdm(
        range(min_frames),
        desc=video_dir.name,
        unit="frames",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    wall_start = time.perf_counter()
    try:
        for _ in iterator:
            # Read all camera frames in parallel — each cap is independent, and
            # cv2 VideoCapture.read() releases the GIL during I/O and decoding.
            _t = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(captures)) as pool:
                fut_map = {cam_id: pool.submit(cap.read) for cam_id, cap in captures.items()}
            images: dict[str, np.ndarray] = {}
            for cam_id, fut in fut_map.items():
                ok, frame = fut.result()
                if not ok or frame is None:
                    logger.warning("Camera %r ran out of frames at frame %d", cam_id, frame_number)
                else:
                    images[cam_id] = frame
            if timer is not None:
                timer.stop("frame_read", _t)
            if len(images) < len(cam_ids):
                break

            timestamp_ms = int(frame_number / fps * 1000)
            observations, states = tracker.process_batch(
                images, frame_number, states, timestamp_ms, timings=timer
            )

            for cam_id, obs in observations.items():
                stores[cam_id].add(obs)
                if cam_id in writers:
                    writers[cam_id].write(annotator.annotate(images[cam_id], obs))

            frame_number += 1
    finally:
        for cap in captures.values():
            cap.release()
        for writer in writers.values():
            writer.release()

    if timer is not None and frame_number > 0:
        total_elapsed = time.perf_counter() - wall_start
        print(timer.report(total_elapsed=total_elapsed, n_frames=frame_number, n_cameras=len(cam_ids)))

    results: dict[str, DataStore] = {}
    for cam_id, store in stores.items():
        output_path = output_dir / f"{cam_id}.{fmt}"
        store.save(output_path, fmt=fmt)
        logger.info("Saved %d frames → %s", frame_number, output_path)
        results[cam_id] = store

    return results
