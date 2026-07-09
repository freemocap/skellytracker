"""Run a live RTMPose wholebody demo from a webcam.

Requires onnxruntime and one of the rtmpose GPU/CPU extras.
Models are downloaded automatically on first run (~100–200 MB).

Usage::

    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.run_demo
    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.run_demo --camera 1
    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.run_demo --model rtmw-x-l_384x288
    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.run_demo --provider cpu
    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.run_demo --no-detector
"""
from __future__ import annotations

import argparse

from skellytracker.core.annotation.keypoint_annotator import (
    ConnectionGroupSchema,
    KeypointAnnotator,
    KeypointAnnotatorConfig,
    StageAnnotationSchema,
)
from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.demo_manager import DemoManager
from skellytracker.core.detectors.keypoint_detectors.rtmpose.rtmpose_keypoint_detector import (
    RTMPoseDetectorConfig,
    RTMPoseKeypointDetector,
)
from skellytracker.core.detectors.object_detectors.yolox.yolox_person_detector import (
    YoloxPersonDetector,
    YoloxPersonDetectorConfig,
)
from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
from skellytracker.core.tracker.tracker import Tracker
from skellytracker.core.sessions.execution_provider_name import ExecutionProviderName


def build_rtmpose_demo(
    model_name: str = "rtmw-x-l_256x192",
    provider: ExecutionProviderName | None = None,
    use_person_detector: bool = True,
    camera_index: int = 0,
) -> DemoManager:
    """Build a DemoManager for the RTMPose wholebody tracker.

    Parameters
    ----------
    model_name:
        RTMPose model to load. One of: ``"rtmw-x-l_256x192"`` (default),
        ``"rtmw-x-l_384x288"`` (slower, more accurate),
        ``"rtmw-l-m_256x192"`` (faster, lighter).
    provider:
        ONNX Runtime execution provider. ``None`` = auto-select
        (CoreML on macOS, CUDA elsewhere, CPU fallback).
    use_person_detector:
        When ``True`` (default), runs YOLOX first to locate the person and
        passes the crop to RTMPose. When ``False``, RTMPose runs on the full
        frame — faster but less accurate when the subject is small.
    camera_index:
        OpenCV camera index (passed to DemoManager, unused here).
    """
    models = [RTMPoseKeypointDetector.model_spec(model_name)]
    if use_person_detector:
        models.insert(0, YoloxPersonDetector.model_spec("yolox-m"))

    session_config = OnnxSessionConfig(
        models=models,
        execution_provider=provider,
    )
    session = OnnxSession.create(session_config)

    object_detector_config = YoloxPersonDetectorConfig() if use_person_detector else None
    stage = DetectionStageConfig(
        name="wholebody",
        object_detector=object_detector_config,
        keypoint_detectors=[RTMPoseDetectorConfig(model_name=model_name)],
    )

    tracker = Tracker.create(
        TrackerConfig(stages=[stage]),
        sessions={"onnx": session},
    )

    groups = RTMPoseKeypointDetector.connection_groups()
    annotator = KeypointAnnotator.create(
        KeypointAnnotatorConfig(stage_schemas={
            "wholebody": StageAnnotationSchema(
                keypoint_color=(0, 255, 128),
                keypoint_radius=3,
                connection_groups=(
                    ConnectionGroupSchema(connections=groups["body"],       connection_color=(0, 200, 100), connection_thickness=1),
                    ConnectionGroupSchema(connections=groups["right_hand"], connection_color=(0, 100, 255), connection_thickness=1),
                    ConnectionGroupSchema(connections=groups["left_hand"],  connection_color=(255, 100, 0), connection_thickness=1),
                    ConnectionGroupSchema(connections=groups["face"],       connection_color=(200, 0, 200), connection_thickness=1),
                ),
            ),
        })
    )

    return DemoManager(tracker=tracker, annotator=annotator, window_title="RTMPose Demo")


def main() -> None:
    parser = argparse.ArgumentParser(description="RTMPose wholebody live demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--model",
        choices=["rtmw-x-l_256x192", "rtmw-x-l_384x288", "rtmw-l-m_256x192"],
        default="rtmw-x-l_256x192",
        help="RTMPose model to use (default: rtmw-x-l_256x192)",
    )
    parser.add_argument(
        "--provider",
        choices=["cuda", "trt", "coreml", "directml", "cpu"],
        default=None,
        help="ONNX Runtime execution provider (default: auto)",
    )
    parser.add_argument(
        "--no-detector",
        action="store_true",
        help="Skip YOLOX person detection; run RTMPose on the full frame",
    )
    args = parser.parse_args()

    demo = build_rtmpose_demo(
        model_name=args.model,
        provider=args.provider,
        use_person_detector=not args.no_detector,
    )
    demo.run_webcam(camera_index=args.camera)


if __name__ == "__main__":
    main()
