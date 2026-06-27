import logging

from skellytracker.old.base_tracker.base_tracker_abcs import BaseRecorder, BaseTracker
from skellytracker.old.composite_gpu_tracker.composite_gpu_annotator import (
    CompositeGPUImageAnnotator,
)
from skellytracker.old.composite_gpu_tracker.composite_gpu_config import (
    CompositeGPUTrackerConfig,
)
from skellytracker.old.composite_gpu_tracker.composite_gpu_detector import (
    CompositeGPUDetector,
)

logger = logging.getLogger(__name__)


class CompositeGPURecorder(BaseRecorder):
    pass


class CompositeGPUTracker(BaseTracker):
    """
    Composable GPU-accelerated tracker for body + hands + face.

    Uses RTMO (one-stage) for body detection + pose (17 keypoints),
    and RTMPose hand/face ONNX models for hands (21×2) and face (68).

    All models run through ONNX Runtime on GPU under a single CUDA context,
    with batched inference support for multi-camera pipelines.

    Usage:
        tracker = CompositeGPUTracker.create()
        observation = tracker.process_image(frame_number=0, image=rgb_image)
        annotated = tracker.annotate_image(image=rgb_image, observation=observation)
    """

    config: CompositeGPUTrackerConfig
    detector: CompositeGPUDetector
    annotator: CompositeGPUImageAnnotator
    recorder: CompositeGPURecorder | None = None

    @classmethod
    def create(cls, config: CompositeGPUTrackerConfig | None = None) -> "CompositeGPUTracker":
        if config is None:
            config = CompositeGPUTrackerConfig()

        detector = CompositeGPUDetector.create(config.detector_config)
        annotator = CompositeGPUImageAnnotator.create(config.annotator_config)

        return cls(
            config=config,
            detector=detector,
            annotator=annotator,
            recorder=CompositeGPURecorder(),
        )


if __name__ == "__main__":
    CompositeGPUTracker.create().demo()
