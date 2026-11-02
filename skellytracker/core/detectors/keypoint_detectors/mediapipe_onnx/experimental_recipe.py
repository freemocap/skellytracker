"""Opt-in experiment composed from the production hybrid recipe, without crop overrides."""

from pathlib import Path

from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.detectors.keypoint_detectors.mediapipe_onnx.hand_detector import (
    OnnxHandDetectorConfig,
)
from skellytracker.core.tracker.tracker_presets import combined_tracker_config


def experimental_tracker_config(
    *, model_directory: Path | None = None, **tracking_settings
) -> TrackerConfig:
    config = combined_tracker_config(**tracking_settings)
    for child in config.stages[0].children:
        (cpu_config,) = child.keypoint_detectors
        child.keypoint_detectors = [
            OnnxHandDetectorConfig(
                assumed_handedness=cpu_config.assumed_handedness,
                min_hand_detection_confidence=cpu_config.min_hand_detection_confidence,
                min_hand_presence_confidence=cpu_config.min_hand_presence_confidence,
                model_directory=model_directory,
            )
        ]
    # The schema and mapping profile remain the shared hybrid preset. The child
    # detector types and model descriptors identify the experimental implementation.
    return config
