"""Run a converted DLC model through the standard Tracker demo framework."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from skellytracker.core.annotation.keypoint_annotator import (
    KeypointAnnotator, KeypointAnnotatorConfig, StageAnnotationSchema,
)
from skellytracker.core.io.demo_manager import DemoManager
from skellytracker.core.tracker.tracker_factory import build_tracker
from skellytracker.core.tracker.tracker_presets import dlc_tracker_config, dlc_tracker_metadata


def build_dlc_demo(
    sidecar_path: str | Path,
    provider: Literal["cpu", "cuda"] = "cpu",
    confidence_threshold: float = 0.4,
) -> DemoManager:
    """Use the artifact's point names, connections and whole-image recipe."""
    metadata = dlc_tracker_metadata(sidecar_path=sidecar_path)
    config = dlc_tracker_config(sidecar_path=sidecar_path, confidence_threshold=confidence_threshold)
    tracker = build_tracker(config, onnx_overrides={"execution_provider": provider, "fp16": False})
    annotator = KeypointAnnotator.create(KeypointAnnotatorConfig(
        confidence_threshold=confidence_threshold,
        stage_schemas={metadata.stage_path[0]: StageAnnotationSchema(connections=metadata.connections)},
    ))
    return DemoManager(tracker=tracker, annotator=annotator, window_title="DLC Keypoint Demo")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DLC webcam/video demo using a converted ONNX sidecar")
    parser.add_argument("--sidecar", required=True, type=Path, help="Converted model YAML; keep its ONNX artifact alongside it")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    source.add_argument("--video", type=Path, help="Read every frame from a video instead of a webcam")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu", help="Explicit FP32 provider (default: cpu)")
    parser.add_argument("--confidence", type=float, default=0.4, help="Keypoint confidence threshold [0, 1]")
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--headless", action="store_true", help="Process --video without opening a window")
    parser.add_argument("--output-video", type=Path, help="Write annotated --video to an MP4 file")
    args = parser.parse_args(argv)
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if (args.headless or args.output_video) and args.video is None:
        parser.error("--headless and --output-video require --video")
    if args.output_video and args.output_video.resolve() == args.video.resolve():
        parser.error("--output-video must differ from the input video")
    demo = build_dlc_demo(args.sidecar, args.provider, args.confidence)
    if args.video:
        count = demo.run_video(args.video, rotation=args.rotate, headless=args.headless, output_path=args.output_video)
        print(f"Processed {count} frames")
    else:
        demo.run_webcam(camera_index=args.camera, rotation=args.rotate)


if __name__ == "__main__":
    main()
