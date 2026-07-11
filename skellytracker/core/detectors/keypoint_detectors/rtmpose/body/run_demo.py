"""Live RTMPose body demo (23 keypoints: COCO body + feet).

Usage::

    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.body.run_demo
    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.body.run_demo --camera 1
    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.body.run_demo --model rtmpose-s_256x192
    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.body.run_demo --provider cpu
    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.body.run_demo --no-detector
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
from skellytracker.core.detectors.keypoint_detectors.rtmpose.body.rtmpose_body_detector import (
    RTMPoseBodyDetector,
    RTMPoseBodyDetectorConfig,
)
from skellytracker.core.detectors.object_detectors.yolox.yolox_person_detector import (
    YoloxPersonDetector,
    YoloxPersonDetectorConfig,
)
from skellytracker.core.sessions.execution_provider_name import ExecutionProviderName
from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
from skellytracker.core.tracker.tracker import Tracker


def build_body_demo(
    model_name: str = "rtmpose-m_256x192",
    provider: ExecutionProviderName | None = None,
    use_person_detector: bool = True,
    camera_index: int = 0,
) -> DemoManager:
    models = [RTMPoseBodyDetector.model_spec(model_name)]
    if use_person_detector:
        models.insert(0, YoloxPersonDetector.model_spec("yolox-m"))

    session = OnnxSession.create(OnnxSessionConfig(batch_size=1, models=models, execution_provider=provider))

    stage = DetectionStageConfig(
        name="body",
        object_detector=YoloxPersonDetectorConfig() if use_person_detector else None,
        keypoint_detectors=[RTMPoseBodyDetectorConfig(model_name=model_name)],
    )
    tracker = Tracker.create(TrackerConfig(stages=[stage]), sessions={"onnx": session})
    annotator = KeypointAnnotator.create(KeypointAnnotatorConfig(stage_schemas={
        "body": StageAnnotationSchema(connections=RTMPoseBodyDetector.connections()),
    }))
    return DemoManager(tracker=tracker, annotator=annotator, window_title="RTMPose Body Demo")


def main() -> None:
    parser = argparse.ArgumentParser(description="RTMPose body live demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--model",
        choices=["rtmpose-m_256x192", "rtmpose-s_256x192"],
        default="rtmpose-m_256x192",
    )
    parser.add_argument("--provider", choices=["cuda", "trt", "coreml", "directml", "cpu"], default=None)
    parser.add_argument("--no-detector", action="store_true")
    args = parser.parse_args()

    demo = build_body_demo(
        model_name=args.model,
        provider=args.provider,
        use_person_detector=not args.no_detector,
    )
    demo.run_webcam(camera_index=args.camera)


if __name__ == "__main__":
    main()
