"""Run YOLOX → RTMPose wholebody on a folder of synchronised camera videos.

Requires onnxruntime (CPU or GPU). Install with:
    uv sync --extra all-cpu       # CPU
    uv sync --extra all-trt       # NVIDIA GPU with TensorRT

Usage:
    python -m skellytracker.examples.run_rtmpose_on_folder \\
        /path/to/synchronized_videos /path/to/output

The number of cameras is detected automatically from the video folder and
passed as batch_size so all cameras share a single GPU call per model per frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import skellytracker.core.detectors.keypoint_detectors.rtmpose  # noqa: F401
import skellytracker.core.detectors.object_detectors.yolox  # noqa: F401
from skellytracker.core import DetectionStageConfig, Tracker, TrackerConfig, process_folder
from skellytracker.core.detectors.keypoint_detectors.rtmpose import RTMPoseDetectorConfig, RTMPoseKeypointDetector
from skellytracker.core.detectors.object_detectors.yolox import YoloxPersonDetector, YoloxPersonDetectorConfig
from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
from skellytracker.core.temporal_processing.temporal_processing_config import (
    BBoxPolicyConfig,
    KeypointsWithinBBoxRatioConfig,
)

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def run_rtmpose_on_folder(
    synchronized_videos_folder: Path,
    output_folder: Path,
    yolox_model: str = "yolox-m",
    rtmpose_model: str = "rtmw-x-l_256x192",
) -> None:
    video_paths = sorted(
        p for p in synchronized_videos_folder.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES
    )
    if not video_paths:
        raise FileNotFoundError(f"No video files found in {synchronized_videos_folder}")

    n_cameras = len(video_paths)
    print(f"Found {n_cameras} camera(s): {[p.name for p in video_paths]}")

    session = OnnxSession.create(OnnxSessionConfig(
        batch_size=n_cameras,
        models=[
            YoloxPersonDetector.model_spec(yolox_model),
            RTMPoseKeypointDetector.model_spec(rtmpose_model),
        ],
    ))

    config = TrackerConfig(
        stages=[
            DetectionStageConfig(
                name="body",
                object_detector=YoloxPersonDetectorConfig(),
                keypoint_detectors=[RTMPoseDetectorConfig()],
                bbox_policy=BBoxPolicyConfig(
                    # Run YOLOX every 5 frames; on skipped frames re-derive the
                    # crop from the previous frame's keypoints so the RTMPose
                    # input tracks the subject without a detector call.
                    redetect_interval=5,
                    keypoint_bbox_expansion=0.2,
                    # Fall back to redetection if too many keypoints drift
                    # outside the current bbox (subject moved suddenly).
                    fitness_checks=[KeypointsWithinBBoxRatioConfig(threshold=0.6)],
                ),
            )
        ]
    )
    tracker = Tracker.create(config, {"onnx": session})

    try:
        results = process_folder(tracker, None, synchronized_videos_folder, output_folder, profile=True)
    finally:
        tracker.close()

    for stem, store in results.items():
        arr = store.to_array()
        print(f"  {stem}: {arr.shape} ({arr.shape[0]} frames, {arr.shape[1]} keypoints)")


if __name__ == "__main__":
    videos_dir = Path("/Users/philipqueen/freemocap_data/recording_sessions/freemocap_test_data/synchronized_videos/")
    output_dir = Path("/Users/philipqueen/freemocap_data/recording_sessions/freemocap_test_data/skellytracker_rtmpose/")

    if len(sys.argv) > 1:
        videos_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])

    run_rtmpose_on_folder(videos_dir, output_dir)
