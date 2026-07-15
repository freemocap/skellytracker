"""Run a live charuco board detection demo from a webcam.

Usage:
    uv run python -m skellytracker.core.detectors.keypoint_detectors.charuco.run_demo
    uv run python -m skellytracker.core.detectors.keypoint_detectors.charuco.run_demo --camera 1
    uv run python -m skellytracker.core.detectors.keypoint_detectors.charuco.run_demo --squares-x 7 --squares-y 5 --square-mm 58
"""
from __future__ import annotations

import argparse

from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.io.demo_manager import DemoManager
from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_annotator import (
    CharucoAnnotatorConfig,
)
from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_board_definition import (
    CharucoBoardDefinition,
)
from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_detector_config import (
    CharucoDetectorConfig,
)
from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_observation_annotator import (
    CharucoObservationAnnotator,
    _CharucoObservationAnnotatorConfig,
)
from skellytracker.core.sessions.cpu_session import CpuSession, CpuSessionConfig
from skellytracker.core.tracker.tracker import Tracker

_STAGE_NAME = "charuco"


def build_charuco_demo(
    board_def: CharucoBoardDefinition | None = None,
    camera_index: int = 0,
) -> DemoManager:
    if board_def is None:
        board_def = CharucoBoardDefinition.create_letter_size_5x3()

    session = CpuSession.create(CpuSessionConfig())
    sessions = {"cpu": session}

    stage = DetectionStageConfig(
        name=_STAGE_NAME,
        keypoint_detectors=[CharucoDetectorConfig(board=board_def)],
    )
    tracker = Tracker.create(TrackerConfig(stages=[stage]), sessions)

    annotator = CharucoObservationAnnotator.create(
        _CharucoObservationAnnotatorConfig(
            board_def=board_def,
            annotator_config=CharucoAnnotatorConfig(),
            stage_name=_STAGE_NAME,
        )
    )

    return DemoManager(tracker=tracker, annotator=annotator, window_title="Charuco Demo")


def main() -> None:
    parser = argparse.ArgumentParser(description="Charuco board live demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--squares-x", type=int, default=5, help="Board columns (default: 5)")
    parser.add_argument("--squares-y", type=int, default=3, help="Board rows (default: 3)")
    parser.add_argument("--square-mm", type=float, default=54.0, help="Square side length in mm (default: 54.0)")
    parser.add_argument("--marker-ratio", type=float, default=0.8, help="ArUco marker size ratio (default: 0.8)")
    args = parser.parse_args()

    board_def = CharucoBoardDefinition(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_mm=args.square_mm,
        marker_length_ratio=args.marker_ratio,
    )

    demo = build_charuco_demo(board_def=board_def, camera_index=args.camera)
    demo.run_webcam(camera_index=args.camera)


if __name__ == "__main__":
    main()
