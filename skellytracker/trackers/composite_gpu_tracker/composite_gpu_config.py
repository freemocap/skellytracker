from typing import Literal

from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseDetectorConfig,
    BaseImageAnnotatorConfig,
    BaseTrackerConfig,
    TrackerType,
)
from skellytracker.trackers.composite_gpu_tracker.composite_gpu_session import (
    CompositeGPUSessionConfig,
)
from skellytracker.trackers.gpu_utils.ort_session_utils import ExecutionProviderName


class CompositeGPUDetectorConfig(BaseDetectorConfig):
    tracker_type: Literal["rtmo_hybrid"] = "rtmo_hybrid"  # type: ignore[assignment]
    confidence_threshold: float = 0.5

    # GPU session config
    session_config: CompositeGPUSessionConfig = Field(default_factory=CompositeGPUSessionConfig)

    # Which sub-detectors to run
    detect_hands: bool = True
    detect_face: bool = True


class CompositeGPUImageAnnotatorConfig(BaseImageAnnotatorConfig):
    show_overlay: bool = False
    show_body: bool = True
    show_hands: bool = True
    show_face: bool = True
    body_kpt_thr: float = 0.3
    hand_kpt_thr: float = 0.3
    face_kpt_thr: float = 0.3


class CompositeGPUTrackerConfig(BaseTrackerConfig):
    detector_config: CompositeGPUDetectorConfig = Field(default_factory=CompositeGPUDetectorConfig)
    annotator_config: CompositeGPUImageAnnotatorConfig = Field(default_factory=CompositeGPUImageAnnotatorConfig)
