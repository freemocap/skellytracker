"""Live RTMPose hand demo (21 keypoints, single hand).

Runs the hand detector on the full frame without upstream cropping. For best
results, hold your hand close to the camera and centred in the frame. In a
production pipeline you would crop individual hand regions from wrist positions
detected by an upstream body detector.

Usage::

    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.hand.run_demo
    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.hand.run_demo --camera 1
    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.hand.run_demo --provider cpu
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
from skellytracker.core.detectors.keypoint_detectors.rtmpose.hand.rtmpose_hand_detector import (
    RTMPoseHandDetector,
    RTMPoseHandDetectorConfig,
)
from skellytracker.core.sessions.execution_provider_name import ExecutionProviderName
from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
from skellytracker.core.tracker.tracker import Tracker


def build_hand_demo(
    provider: ExecutionProviderName | None = None,
    camera_index: int = 0,
) -> DemoManager:
    session = OnnxSession.create(OnnxSessionConfig(
        models=[RTMPoseHandDetector.model_spec("rtmpose-m_256x256")],
        execution_provider=provider,
    ))
    stage = DetectionStageConfig(
        name="hand",
        keypoint_detectors=[RTMPoseHandDetectorConfig()],
    )
    tracker = Tracker.create(TrackerConfig(stages=[stage]), sessions={"onnx": session})
    annotator = KeypointAnnotator.create(KeypointAnnotatorConfig(stage_schemas={
        "hand": StageAnnotationSchema(connections=RTMPoseHandDetector.connections()),
    }))
    return DemoManager(tracker=tracker, annotator=annotator, window_title="RTMPose Hand Demo")


def main() -> None:
    parser = argparse.ArgumentParser(description="RTMPose hand live demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--provider", choices=["cuda", "trt", "coreml", "directml", "cpu"], default=None)
    args = parser.parse_args()

    demo = build_hand_demo(provider=args.provider)
    demo.run_webcam(camera_index=args.camera)


if __name__ == "__main__":
    main()
