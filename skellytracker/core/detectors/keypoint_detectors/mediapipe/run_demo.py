"""Run a live mediapipe pose/hand/face demo from a webcam.

Usage:
    uv run python -m skellytracker.core.detectors.keypoint_detectors.mediapipe.run_demo
    uv run python -m skellytracker.core.detectors.keypoint_detectors.mediapipe.run_demo --camera 1
    uv run python -m skellytracker.core.detectors.keypoint_detectors.mediapipe.run_demo --no-hands --no-face
"""
from __future__ import annotations

import argparse

from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.io.demo_manager import DemoManager
from skellytracker.core.annotation.keypoint_annotator import (
    ConnectionGroupSchema,
    KeypointAnnotator,
    KeypointAnnotatorConfig,
    StageAnnotationSchema,
)
from skellytracker.core.tracker.tracker import Tracker
from skellytracker.core.detectors.keypoint_detectors.mediapipe.face.mediapipe_face_detector import MediapipeFaceDetectorConfig, MediapipeFaceKeypointDetector
from skellytracker.core.detectors.keypoint_detectors.mediapipe.hands.mediapipe_hand_detector import MediapipeHandDetectorConfig, MediapipeHandKeypointDetector
from skellytracker.core.detectors.keypoint_detectors.mediapipe.mediapipe_model_manager import MediapipePoseModelComplexity
from skellytracker.core.detectors.keypoint_detectors.mediapipe.body.mediapipe_pose_detector import MediapipePoseDetectorConfig, MediapipePoseKeypointDetector
from skellytracker.core.detectors.object_detectors.keypoint_bbox import KeypointBoundingBoxDetectorConfig
from skellytracker.core.sessions.cpu_session import CpuSession, CpuSessionConfig
from skellytracker.core.temporal_processing.temporal_processing_config import BBoxSmoothingConfig
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

    keypoint_detectors: list = []
    connection_groups: list[ConnectionGroupSchema] = []

    if detect_pose:
        keypoint_detectors.append(MediapipePoseDetectorConfig(model_complexity=pose_complexity))
        connection_groups.append(ConnectionGroupSchema(
            connections=MediapipePoseKeypointDetector.connections(),
            connection_color=(0, 180, 0),
        ))
    if detect_hands:
        keypoint_detectors.append(MediapipeHandDetectorConfig())
        connection_groups.append(ConnectionGroupSchema(
            connections=MediapipeHandKeypointDetector.connections(),
            connection_color=(200, 80, 0),
            keypoint_color=(200, 80, 0),
        ))
    if detect_face:
        keypoint_detectors.append(MediapipeFaceDetectorConfig())
        connection_groups.append(ConnectionGroupSchema(
            connections=MediapipeFaceKeypointDetector.connections(),
            connection_color=(0, 160, 200),
            connection_thickness=1,
            keypoint_color=(0, 160, 200),
        ))

    stage = DetectionStageConfig(name="composite", keypoint_detectors=keypoint_detectors)
    tracker = Tracker.create(TrackerConfig(stages=[stage]), sessions)

    annotator = KeypointAnnotator.create(
        KeypointAnnotatorConfig(stage_schemas={
            "composite": StageAnnotationSchema(
                keypoint_color=(0, 255, 0),
                keypoint_radius=3,
                connection_groups=tuple(connection_groups),
            ),
        })
    )

    return DemoManager(tracker=tracker, annotator=annotator, window_title="MediaPipe Demo")


def _hand_child_stage(side: str) -> DetectionStageConfig:
    """A hand DetectionStage cropped from the parent body stage's own wrist/
    index/pinky keypoints via the generic KeypointBoundingBoxDetector.

    This is backend-agnostic: KeypointBoundingBoxDetector only reads named
    keypoints off whatever parent stage it's attached to, so the same crop
    mechanism works with any body/hand detector pair (RTMPose, etc.) that
    exposes these point names — not just MediaPipe.
    """
    return DetectionStageConfig(
        name=f"{side}_hand",
        object_detector=KeypointBoundingBoxDetectorConfig(
            center_keypoint_names=(f"{side}_index", f"{side}_pinky"),
            scale_keypoint_pairs=(
                (f"{side}_wrist", f"{side}_index"),
                (f"{side}_wrist", f"{side}_pinky"),
            ),
            scale_factor=3.0,
            min_box_size_px=120.0,
        ),
        # assumed_handedness overrides MediaPipe's own per-hand label — the
        # crop is already dedicated to this side (it was cropped from this
        # side's wrist/index/pinky), which is more reliable than MediaPipe's
        # label on a tight single-hand crop. See mediapipe_hand_detector.py.
        keypoint_detectors=[MediapipeHandDetectorConfig(num_hands=1, assumed_handedness=side)],
        # The box is re-derived from the body stage's wrist/index/pinky
        # estimate every frame (redetect_interval=1, the default) with no
        # temporal filtering of its own — EMA smoothing here damps
        # frame-to-frame jitter in that estimate (most noticeable right as
        # the hand nears the frame edge, where body-level landmark accuracy
        # degrades) so the crop doesn't jump/shrink erratically.
        bbox_smoothing=BBoxSmoothingConfig(alpha=0.4),
    )


def build_mediapipe_demo_with_cropped_hands(
    pose_complexity: MediapipePoseModelComplexity = MediapipePoseModelComplexity.LITE,
) -> DemoManager:
    """Body stage with hands run as child stages cropped from its own wrist/
    index/pinky keypoints, instead of MediaPipe HandLandmarker scanning the
    full frame — much better detection rate when hands are small in frame.
    """
    session = MediaPipeSession.create(MediaPipeSessionConfig())
    cpu_session = CpuSession.create(CpuSessionConfig())

    body_stage = DetectionStageConfig(
        name="body",
        keypoint_detectors=[MediapipePoseDetectorConfig(model_complexity=pose_complexity)],
        children=[_hand_child_stage("left"), _hand_child_stage("right")],
    )
    tracker = Tracker.create(
        TrackerConfig(stages=[body_stage]),
        {"mediapipe": session, "cpu": cpu_session},
    )

    annotator = KeypointAnnotator.create(
        KeypointAnnotatorConfig(stage_schemas={
            "body": StageAnnotationSchema(
                keypoint_color=(0, 255, 0),
                keypoint_radius=3,
                connection_groups=(
                    ConnectionGroupSchema(
                        connections=MediapipePoseKeypointDetector.connections(),
                        connection_color=(0, 180, 0),
                    ),
                ),
            ),
            "left_hand": StageAnnotationSchema(
                draw_boxes=True,
                box_color_detected=(200, 80, 0),
                box_color_reused=(200, 160, 0),
                connection_groups=(
                    ConnectionGroupSchema(
                        connections=MediapipeHandKeypointDetector.connections(),
                        connection_color=(200, 80, 0),
                        keypoint_color=(200, 80, 0),
                    ),
                ),
            ),
            "right_hand": StageAnnotationSchema(
                draw_boxes=True,
                box_color_detected=(0, 80, 200),
                box_color_reused=(0, 160, 200),
                connection_groups=(
                    ConnectionGroupSchema(
                        connections=MediapipeHandKeypointDetector.connections(),
                        connection_color=(0, 80, 200),
                        keypoint_color=(0, 80, 200),
                    ),
                ),
            ),
        })
    )

    return DemoManager(
        tracker=tracker, annotator=annotator, window_title="MediaPipe Demo — Cropped Hands"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MediaPipe live demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--no-pose", action="store_true", help="Disable body pose detection")
    parser.add_argument("--no-hands", action="store_true", help="Disable hand detection")
    parser.add_argument("--no-face", action="store_true", help="Disable face detection")
    parser.add_argument(
        "--cropped-hands",
        action="store_true",
        help=(
            "Run hands as child stages cropped from the body stage's own "
            "wrist/index/pinky keypoints (KeypointBoundingBoxDetector) "
            "instead of scanning the full frame. Implies --no-face."
        ),
    )
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

    if args.cropped_hands:
        demo = build_mediapipe_demo_with_cropped_hands(
            pose_complexity=complexity_map[args.complexity],
        )
    else:
        demo = build_mediapipe_demo(
            detect_pose=not args.no_pose,
            detect_hands=not args.no_hands,
            detect_face=not args.no_face,
            pose_complexity=complexity_map[args.complexity],
        )
    demo.run_webcam(camera_index=args.camera)


if __name__ == "__main__":
    main()
