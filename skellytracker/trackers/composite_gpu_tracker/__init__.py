"""Composable GPU-accelerated tracker for body + hands + face.

Uses selectable ONNX models for each body part:
- Body: RTMO (one-stage, 17 COCO keypoints)
- Hands: RTMPose hand ONNX (21 keypoints per hand, crop from body wrists)
- Face: RTMPose face ONNX (106 LaPa keypoints, crop from body head landmarks)

All models share a single ONNX Runtime CUDA context with batched inference.
"""

from skellytracker.trackers.composite_gpu_tracker.__composite_gpu_tracker import (
    CompositeGPUTracker,
)
from skellytracker.trackers.composite_gpu_tracker.composite_gpu_config import (
    CompositeGPUTrackerConfig,
)
from skellytracker.trackers.composite_gpu_tracker.composite_gpu_session import (
    CompositeGPUSession,
    CompositeGPUSessionConfig,
)
from skellytracker.trackers.composite_gpu_tracker.composite_gpu_detector import (
    CompositeGPUDetector,
)
from skellytracker.trackers.composite_gpu_tracker.composite_gpu_observation import (
    CompositeGPUObservation,
)
from skellytracker.trackers.composite_gpu_tracker.sub_model_spec import (
    SubModelSpec,
    TrackerPreset,
)

__all__ = [
    "CompositeGPUTracker",
    "CompositeGPUTrackerConfig",
    "CompositeGPUSession",
    "CompositeGPUSessionConfig",
    "CompositeGPUDetector",
    "CompositeGPUObservation",
    "SubModelSpec",
    "TrackerPreset",
]
