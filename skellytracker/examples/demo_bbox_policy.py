"""Live webcam demo with a configurable bbox policy and boxes drawn on screen.

The redetect cadence and keypoint-bbox expansion here are tuned to match the
old (skellytracker/old/rtmpose_tracker) YOLOX-skip tracking behavior: YOLOX
re-runs roughly every ``redetect_seconds`` seconds (old default 5s), and in
between frames the crop is a *tight* expansion (old default 0.05, not 0.2)
of the previous frame's keypoints, smoothed with an EMA so the crop doesn't
snap between the detector's box and the keypoint-derived box.

Usage:
    uv run python -m skellytracker.examples.demo_bbox_policy
    uv run python -m skellytracker.examples.demo_bbox_policy --redetect-seconds 5 --keypoint-bbox-expansion 0.05
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
from skellytracker.core.io.demo_manager import DemoManager
from skellytracker.core.detectors.keypoint_detectors.rtmpose.rtmpose_keypoint_detector import (
    RTMPoseDetectorConfig,
    RTMPoseKeypointDetector,
)
from skellytracker.core.detectors.object_detectors.yolox.yolox_person_detector import (
    YoloxPersonDetector,
    YoloxPersonDetectorConfig,
)
from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
from skellytracker.core.temporal_processing.temporal_processing_config import (
    BBoxPolicyConfig,
    BBoxSmoothingConfig,
    KeypointsWithinBBoxRatioConfig,
)
from skellytracker.core.tracker.tracker import Tracker

# Webcams in this demo run close enough to 30fps that a frame-count interval
# approximates old's wall-clock ``min_detection_interval_seconds``. The core
# BBoxPolicy is frame-based (see should_redetect in bbox_policy.py), so we
# convert seconds -> frames here rather than in the core policy.
_ASSUMED_FPS = 30.0


def build_demo(
    model_name: str = "rtmw-x-l_256x192",
    redetect_seconds: float = 5.0,
    keypoint_bbox_expansion: float | None = 0.05,
    within_bbox_ratio_threshold: float = 0.5,
    bbox_smoothing_alpha: float = 0.4,
) -> DemoManager:
    models = [YoloxPersonDetector.model_spec("yolox-m"), RTMPoseKeypointDetector.model_spec(model_name)]
    session = OnnxSession.create(OnnxSessionConfig(batch_size=1, models=models))

    redetect_interval = max(1, round(redetect_seconds * _ASSUMED_FPS))

    stage = DetectionStageConfig(
        name="wholebody",
        object_detector=YoloxPersonDetectorConfig(),
        keypoint_detectors=[RTMPoseDetectorConfig(model_name=model_name)],
        bbox_policy=BBoxPolicyConfig(
            redetect_interval=redetect_interval,
            keypoint_bbox_expansion=keypoint_bbox_expansion,
            fitness_checks=[KeypointsWithinBBoxRatioConfig(threshold=within_bbox_ratio_threshold)],
        ),
        bbox_smoothing=BBoxSmoothingConfig(alpha=bbox_smoothing_alpha),
    )

    tracker = Tracker.create(TrackerConfig(stages=[stage]), sessions={"onnx": session})

    groups = RTMPoseKeypointDetector.connection_groups()
    annotator = KeypointAnnotator.create(
        KeypointAnnotatorConfig(stage_schemas={
            "wholebody": StageAnnotationSchema(
                keypoint_color=(0, 255, 128),
                keypoint_radius=3,
                draw_boxes=True,
                connection_groups=(
                    ConnectionGroupSchema(connections=groups["body"],       connection_color=(0, 200, 100), connection_thickness=1),
                    ConnectionGroupSchema(connections=groups["right_hand"], connection_color=(0, 100, 255), connection_thickness=1),
                    ConnectionGroupSchema(connections=groups["left_hand"],  connection_color=(255, 100, 0), connection_thickness=1),
                    ConnectionGroupSchema(connections=groups["face"],       connection_color=(200, 0, 200), connection_thickness=1),
                ),
            ),
        })
    )

    return DemoManager(tracker=tracker, annotator=annotator, window_title="BBox Policy Demo")


def main() -> None:
    parser = argparse.ArgumentParser(description="RTMPose live demo with bbox policy + boxes drawn")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--redetect-seconds", type=float, default=5.0)
    parser.add_argument("--keypoint-bbox-expansion", type=float, default=0.05)
    parser.add_argument("--within-bbox-ratio-threshold", type=float, default=0.5)
    parser.add_argument("--bbox-smoothing-alpha", type=float, default=0.4)
    args = parser.parse_args()

    demo = build_demo(
        redetect_seconds=args.redetect_seconds,
        keypoint_bbox_expansion=args.keypoint_bbox_expansion,
        within_bbox_ratio_threshold=args.within_bbox_ratio_threshold,
        bbox_smoothing_alpha=args.bbox_smoothing_alpha,
    )
    demo.run_webcam(camera_index=args.camera)


if __name__ == "__main__":
    main()
