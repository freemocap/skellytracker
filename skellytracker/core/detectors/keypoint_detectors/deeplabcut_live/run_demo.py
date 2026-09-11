"""Run a live DeepLabCut-Live (PyTorch) demo from a webcam.

Requires the deeplabcut-live[pytorch] extra and a PyTorch-exported model
(a single .pt file produced by deeplabcut.export_model). There is no
bundled/default model — you must supply --model-path.

Keypoint names are read from the model itself once loaded, and no skeleton
is declared anywhere, so this demo annotates points only (no connecting
lines), unlike the other detector demos.

Usage::

    uv run python -m skellytracker.core.detectors.keypoint_detectors.deeplabcut_live.run_demo --model-path /path/to/model.pt
    uv run python -m skellytracker.core.detectors.keypoint_detectors.deeplabcut_live.run_demo --model-path /path/to/model.pt --camera 1
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
from skellytracker.core.detectors.keypoint_detectors.deeplabcut_live.dlc_live_keypoint_detector import (
    DLCLiveKeypointDetectorConfig,
)
from skellytracker.core.io.demo_manager import DemoManager
from skellytracker.core.sessions.dlc_live_session import (
    DLCLiveSession,
    DLCLiveSessionConfig,
)
from skellytracker.core.tracker.tracker import Tracker


def build_dlclive_demo(model_path: str, device: str | None = None) -> DemoManager:
    """Build a DemoManager for a DeepLabCut-Live (PyTorch) tracker.

    Runs on the full frame (no ObjectDetector). Pass an ObjectDetectorConfig
    to DetectionStageConfig separately if top-down cropping is desired.
    """
    detector_config = DLCLiveKeypointDetectorConfig(model_path=model_path)
    session = DLCLiveSession.create(DLCLiveSessionConfig(device=device))

    stage = DetectionStageConfig(
        name="dlclive",
        keypoint_detectors=[detector_config],
    )
    tracker = Tracker.create(
        TrackerConfig(stages=[stage]), sessions={"dlclive": session}
    )

    # No skeleton is drawn: connections aren't declared anywhere for a
    # custom-trained DLC-Live model, so points are annotated without lines.
    annotator = KeypointAnnotator.create(
        KeypointAnnotatorConfig(
            stage_schemas={
                "dlclive": StageAnnotationSchema(connections=()),
            }
        )
    )

    return DemoManager(
        tracker=tracker, annotator=annotator, window_title="DeepLabCut-Live Demo"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepLabCut-Live (PyTorch) live demo")
    parser.add_argument(
        "--model-path", required=True, help="Path to an exported PyTorch .pt model"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device, e.g. 'cpu' or 'cuda:0' (default: auto)",
    )
    args = parser.parse_args()

    demo = build_dlclive_demo(model_path=args.model_path, device=args.device)
    demo.run_webcam(camera_index=args.camera)


if __name__ == "__main__":
    main()
