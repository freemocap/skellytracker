"""Run a live mediapipe pose/hand/face demo from a webcam.

Usage:
    uv run python -m skellytracker.core.detectors.mediapipe.run_demo
    uv run python -m skellytracker.core.detectors.mediapipe.run_demo --camera 1
    uv run python -m skellytracker.core.detectors.mediapipe.run_demo --no-hands --no-face
"""
from __future__ import annotations

import argparse

from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.demo_manager import DemoManager
from skellytracker.core.annotation.keypoint_annotator import (
    KeypointAnnotator,
    KeypointAnnotatorConfig,
    StageAnnotationSchema,
)
from skellytracker.core.tracker.tracker import Tracker
from skellytracker.core.detectors.mediapipe.face.mediapipe_face_detector import MediapipeFaceDetectorConfig, MediapipeFaceKeypointDetector
from skellytracker.core.detectors.mediapipe.hands.mediapipe_hand_detector import MediapipeHandDetectorConfig, MediapipeHandKeypointDetector
from skellytracker.core.detectors.mediapipe.mediapipe_model_manager import MediapipePoseModelComplexity
from skellytracker.core.detectors.mediapipe.body.mediapipe_pose_detector import MediapipePoseDetectorConfig, MediapipePoseKeypointDetector
from skellytracker.core.sessions.mediapipe_session import (
    MediaPipeSession,
    MediaPipeSessionConfig,
)


def build_mediapipe_demo(
    detect_pose: bool = True,
    detect_hands: bool = True,
    detect_face: bool = True,
    pose_complexity: MediapipePoseModelComplexity = MediapipePoseModelComplexity.LITE,
) -> DemoManager:
    session = MediaPipeSession.create(MediaPipeSessionConfig())
    sessions = {"mediapipe": session}

    stages: list[DetectionStageConfig] = []
    stage_schemas: dict[str, StageAnnotationSchema] = {}

    if detect_pose:
        stages.append(DetectionStageConfig(
            name="body",
            keypoint_detectors=[MediapipePoseDetectorConfig(model_complexity=pose_complexity)],
        ))
        stage_schemas["body"] = StageAnnotationSchema(
            connections=MediapipePoseKeypointDetector.connections(),
            keypoint_color=(0, 255, 0),
            connection_color=(0, 180, 0),
        )
    if detect_hands:
        stages.append(DetectionStageConfig(
            name="hands",
            keypoint_detectors=[MediapipeHandDetectorConfig()],
        ))
        stage_schemas["hands"] = StageAnnotationSchema(
            connections=MediapipeHandKeypointDetector.connections(),
            keypoint_color=(255, 100, 0),
            connection_color=(200, 80, 0),
        )
    if detect_face:
        stages.append(DetectionStageConfig(
            name="face",
            keypoint_detectors=[MediapipeFaceDetectorConfig()],
        ))
        stage_schemas["face"] = StageAnnotationSchema(
            connections=MediapipeFaceKeypointDetector.connections(),
            keypoint_color=(0, 200, 255),
            connection_color=(0, 160, 200),
            keypoint_radius=2,
            connection_thickness=1,
        )

    tracker = Tracker.create(TrackerConfig(stages=stages), sessions)

    annotator = KeypointAnnotator.create(
        KeypointAnnotatorConfig(stage_schemas=stage_schemas)
    )

    return DemoManager(tracker=tracker, annotator=annotator, window_title="MediaPipe Demo")


def main() -> None:
    parser = argparse.ArgumentParser(description="MediaPipe live demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--no-pose", action="store_true", help="Disable body pose detection")
    parser.add_argument("--no-hands", action="store_true", help="Disable hand detection")
    parser.add_argument("--no-face", action="store_true", help="Disable face detection")
    parser.add_argument(
        "--complexity",
        choices=["lite", "full", "heavy"],
        default="lite",
        help="Pose model complexity (default: lite)",
    )
    args = parser.parse_args()

    complexity_map = {
        "lite": MediapipePoseModelComplexity.LITE,
        "full": MediapipePoseModelComplexity.FULL,
        "heavy": MediapipePoseModelComplexity.HEAVY,
    }

    demo = build_mediapipe_demo(
        detect_pose=not args.no_pose,
        detect_hands=not args.no_hands,
        detect_face=not args.no_face,
        pose_complexity=complexity_map[args.complexity],
    )
    demo.run_webcam(camera_index=args.camera)


if __name__ == "__main__":
    main()
