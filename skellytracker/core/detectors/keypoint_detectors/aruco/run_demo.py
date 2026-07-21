"""Run a live ArUco marker detection demo from a webcam.

Usage:
    uv run python -m skellytracker.core.detectors.keypoint_detectors.aruco.run_demo
    uv run python -m skellytracker.core.detectors.keypoint_detectors.aruco.run_demo --camera 1
    uv run python -m skellytracker.core.detectors.keypoint_detectors.aruco.run_demo --ids 0 1 2 3
"""
from __future__ import annotations

import argparse

import cv2

from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.io.demo_manager import DemoManager
from skellytracker.core.detectors.keypoint_detectors.aruco.aruco_annotator import (
    ArucoAnnotatorConfig,
)
from skellytracker.core.detectors.keypoint_detectors.aruco.aruco_detector_config import (
    ArucoDetectorConfig,
)
from skellytracker.core.detectors.keypoint_detectors.aruco.aruco_observation_annotator import (
    ArucoObservationAnnotator,
    _ArucoObservationAnnotatorConfig,
)
from skellytracker.core.sessions.cpu_session import CpuSession, CpuSessionConfig
from skellytracker.core.tracker.tracker import Tracker

_STAGE_NAME = "aruco"


def build_aruco_demo(
    aruco_ids: tuple[int, ...] = (0, 1, 2, 3),
    aruco_dictionary_enum: int = cv2.aruco.DICT_4X4_50,
    camera_index: int = 0,
) -> DemoManager:
    session = CpuSession.create(CpuSessionConfig())
    sessions = {"cpu": session}

    detector_config = ArucoDetectorConfig(
        aruco_ids=aruco_ids,
        aruco_dictionary_enum=aruco_dictionary_enum,
    )
    stage = DetectionStageConfig(
        name=_STAGE_NAME,
        keypoint_detectors=[detector_config],
    )
    tracker = Tracker.create(TrackerConfig(stages=[stage]), sessions)

    annotator = ArucoObservationAnnotator.create(
        _ArucoObservationAnnotatorConfig(
            aruco_ids=aruco_ids,
            annotator_config=ArucoAnnotatorConfig(),
            stage_name=_STAGE_NAME,
        )
    )

    return DemoManager(tracker=tracker, annotator=annotator, window_title="Aruco Demo")


def main() -> None:
    parser = argparse.ArgumentParser(description="ArUco marker live demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="ArUco marker IDs to track (default: 0 1 2 3)",
    )
    parser.add_argument(
        "--dict",
        type=int,
        default=cv2.aruco.DICT_4X4_50,
        help="ArUco dictionary enum value (default: DICT_4X4_50)",
    )
    args = parser.parse_args()

    demo = build_aruco_demo(
        aruco_ids=tuple(args.ids),
        aruco_dictionary_enum=args.dict,
        camera_index=args.camera,
    )
    demo.run_webcam(camera_index=args.camera)


if __name__ == "__main__":
    main()
