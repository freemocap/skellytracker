"""Backend-free config for the RTMPose detector.

Separated from `rtmpose_detector` (which imports onnxruntime and the ORT session)
so the config can be imported — e.g. to build the SkeletonDetectorConfig union or
type a pipeline config — without loading the ONNX Runtime native library.
"""
from typing import Literal

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetectorConfig, TrackerType
from skellytracker.utilities.gpu_utils.execution_provider_name import ExecutionProviderName

# Backwards-compatible alias maintained for existing callers / configs that
# still pass `device="cuda"`. New code should use `execution_provider`.
_DEVICE_TO_PROVIDER: dict[str, ExecutionProviderName] = {
    "cuda": "cuda",
    "trt": "trt",
    "tensorrt": "trt",
    "mps": "coreml",
    "coreml": "coreml",
    "cpu": "cpu",
}


def _default_execution_provider() -> ExecutionProviderName:
    import sys
    return "coreml" if sys.platform == "darwin" else "cuda"


class RTMPoseDetectorConfig(BaseDetectorConfig):
    tracker_type: Literal[TrackerType.RTMPOSE] = TrackerType.RTMPOSE
    confidence_threshold: float = 0.005
    mode: str = "lightweight"
    backend: str = "onnxruntime"
    device: str = "auto"
    # When set, takes precedence over `device`. Drives the actual ORT provider selection.
    execution_provider: ExecutionProviderName | None = None
    # Which GPU to use. None = auto-select the device with the most VRAM at session creation.
    device_id: int | None = None
    # Keep only the N highest-confidence YOLOX detections. None = keep all.
    # Set to 1 for single-person use to suppress false positives from background clutter.
    max_persons: int | None = 1

    def resolved_provider(self) -> ExecutionProviderName:
        if self.execution_provider is not None:
            return self.execution_provider
        return _default_execution_provider()
