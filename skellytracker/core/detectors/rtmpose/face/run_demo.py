"""Live RTMPose face demo (68 keypoints, iBUG 300-W convention).

Runs the face detector on the full frame without upstream cropping. For best
results, face the camera directly and keep your face centred in the frame. In
a production pipeline you would crop the face region from a bounding box
produced by an upstream person or face detector.

Usage::

    uv run python -m skellytracker.core.detectors.rtmpose.face.run_demo
    uv run python -m skellytracker.core.detectors.rtmpose.face.run_demo --camera 1
    uv run python -m skellytracker.core.detectors.rtmpose.face.run_demo --provider cpu
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
from skellytracker.core.detectors.rtmpose.face.rtmpose_face_detector import (
    RTMPoseFaceDetector,
    RTMPoseFaceDetectorConfig,
)
from skellytracker.core.sessions.execution_provider_name import ExecutionProviderName
from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
from skellytracker.core.tracker.tracker import Tracker


def build_face_demo(
    provider: ExecutionProviderName | None = None,
    camera_index: int = 0,
) -> DemoManager:
    session = OnnxSession.create(OnnxSessionConfig(
        models=[RTMPoseFaceDetector.model_spec("rtmpose-m_256x256")],
        execution_provider=provider,
    ))
    stage = DetectionStageConfig(
        name="face",
        keypoint_detectors=[RTMPoseFaceDetectorConfig()],
    )
    tracker = Tracker.create(TrackerConfig(stages=[stage]), sessions={"onnx": session})
    annotator = KeypointAnnotator.create(KeypointAnnotatorConfig(stage_schemas={
        "face": StageAnnotationSchema(connections=RTMPoseFaceDetector.connections()),
    }))
    return DemoManager(tracker=tracker, annotator=annotator, window_title="RTMPose Face Demo")


def main() -> None:
    parser = argparse.ArgumentParser(description="RTMPose face live demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--provider", choices=["cuda", "trt", "coreml", "directml", "cpu"], default=None)
    args = parser.parse_args()

    demo = build_face_demo(provider=args.provider)
    demo.run_webcam(camera_index=args.camera)


if __name__ == "__main__":
    main()
