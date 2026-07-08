"""Run a live YOLOX person-detection demo from a webcam.

Draws bounding boxes around detected people in real time.
Models are downloaded automatically on first run (~200 MB).

Usage::

    uv run python -m skellytracker.core.detectors.object_detectors.yolox.run_demo
    uv run python -m skellytracker.core.detectors.object_detectors.yolox.run_demo --camera 1
    uv run python -m skellytracker.core.detectors.object_detectors.yolox.run_demo --model yolox-tiny
    uv run python -m skellytracker.core.detectors.object_detectors.yolox.run_demo --provider cpu
    uv run python -m skellytracker.core.detectors.object_detectors.yolox.run_demo --max-detections 5
"""
from __future__ import annotations

import argparse

from skellytracker.core.annotation.keypoint_annotator import (
    KeypointAnnotator,
    KeypointAnnotatorConfig,
    StageAnnotationSchema,
)
from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.demo_manager import DemoManager
from skellytracker.core.detectors.object_detectors.yolox.yolox_person_detector import (
    YoloxPersonDetector,
    YoloxPersonDetectorConfig,
)
from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
from skellytracker.core.tracker.tracker import Tracker
from skellytracker.core.sessions.execution_provider_name import ExecutionProviderName


def build_yolox_demo(
    model_name: str = "yolox-m",
    provider: ExecutionProviderName | None = None,
    score_threshold: float = 0.7,
    max_detections: int | None = 1,
) -> DemoManager:
    """Build a DemoManager that runs YOLOX person detection and draws bounding boxes.

    Parameters
    ----------
    model_name:
        Which YOLOX checkpoint to use: ``"yolox-m"`` (default) or ``"yolox-tiny"``.
    provider:
        ONNX Runtime execution provider. ``None`` = auto-select
        (CoreML on macOS, CUDA elsewhere, CPU fallback).
    score_threshold:
        Minimum confidence to display a detection box.
    max_detections:
        Maximum number of boxes to show per frame. ``None`` = show all.
    """
    session_config = OnnxSessionConfig(
        models=[YoloxPersonDetector.model_spec(model_name)],
        execution_provider=provider,
    )
    session = OnnxSession.create(session_config)

    stage = DetectionStageConfig(
        name="person",
        object_detector=YoloxPersonDetectorConfig(
            model_name=model_name,
            score_threshold=score_threshold,
            max_detections=max_detections,
        ),
        keypoint_detectors=[],
    )

    tracker = Tracker.create(
        TrackerConfig(stages=[stage]),
        sessions={"onnx": session},
    )

    annotator = KeypointAnnotator.create(
        KeypointAnnotatorConfig(stage_schemas={
            "person": StageAnnotationSchema(
                connections=(),
                draw_boxes=True,
                box_color=(0, 200, 255),
            ),
        })
    )

    return DemoManager(tracker=tracker, annotator=annotator, window_title="YOLOX Person Detector Demo")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOX person detection live demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--model",
        choices=["yolox-m", "yolox-tiny"],
        default="yolox-m",
        help="YOLOX model variant (default: yolox-m)",
    )
    parser.add_argument(
        "--provider",
        choices=["cuda", "trt", "coreml", "directml", "cpu"],
        default=None,
        help="ONNX Runtime execution provider (default: auto)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.7,
        help="Minimum detection confidence to display (default: 0.7)",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=1,
        help="Max boxes per frame; 0 = show all (default: 1)",
    )
    args = parser.parse_args()

    max_det = None if args.max_detections == 0 else args.max_detections
    demo = build_yolox_demo(
        model_name=args.model,
        provider=args.provider,
        score_threshold=args.score_threshold,
        max_detections=max_det,
    )
    demo.run_webcam(camera_index=args.camera)


if __name__ == "__main__":
    main()
