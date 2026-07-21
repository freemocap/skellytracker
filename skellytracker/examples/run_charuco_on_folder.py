"""Run the Charuco board detector on a folder of synchronised camera videos.

CPU-only, no model loading (OpenCV Aruco/Charuco detection) — this makes the
per-frame keypoint-detection cost lightweight relative to typical GPU pose
models, which is useful for isolating overhead elsewhere in the pipeline
(e.g. thread-pool churn in DetectionStage.run_batch) since inference itself
won't dominate the profiled timings.

Usage:
    python -m skellytracker.examples.run_charuco_on_folder \\
        /path/to/synchronized_videos /path/to/output
"""
from __future__ import annotations

import sys
from pathlib import Path

import skellytracker.core.detectors.keypoint_detectors.charuco  # noqa: F401
from skellytracker.core import DetectionStageConfig, Tracker, TrackerConfig, process_folder
from skellytracker.core.detectors.keypoint_detectors.charuco import (
    CharucoBoardDefinition,
    CharucoDetectorConfig,
)
from skellytracker.core.sessions.cpu_session import CpuSession, CpuSessionConfig

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def run_charuco_on_folder(
    synchronized_videos_folder: Path,
    output_folder: Path,
) -> None:
    video_paths = sorted(
        p for p in synchronized_videos_folder.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES
    )
    if not video_paths:
        raise FileNotFoundError(f"No video files found in {synchronized_videos_folder}")

    n_cameras = len(video_paths)
    print(f"Found {n_cameras} camera(s): {[p.name for p in video_paths]}")

    session = CpuSession.create(CpuSessionConfig())

    config = TrackerConfig(
        stages=[
            DetectionStageConfig(
                name="charuco",
                keypoint_detectors=[
                    CharucoDetectorConfig(board=CharucoBoardDefinition.create_test_data_7x5())
                ],
            )
        ]
    )
    tracker = Tracker.create(config, {"cpu": session})

    try:
        results = process_folder(tracker, None, synchronized_videos_folder, output_folder, profile=True)
    finally:
        tracker.close()

    for stem, store in results.items():
        arr = store.to_array()
        print(f"  {stem}: {arr.shape} ({arr.shape[0]} frames, {arr.shape[1]} keypoints)")


if __name__ == "__main__":
    videos_dir = Path("/Users/philipqueen/freemocap_data/recording_sessions/freemocap_test_data/synchronized_videos/")
    output_dir = Path("/Users/philipqueen/freemocap_data/recording_sessions/freemocap_test_data/skellytracker_charuco/")

    if len(sys.argv) > 1:
        videos_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])

    run_charuco_on_folder(videos_dir, output_dir)
