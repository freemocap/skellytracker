import logging

from skellytracker.old.base_tracker.base_tracker_abcs import BaseRecorder, BaseTracker
from skellytracker.old.mediapipe_tracker.composite.mediapipe_composite_annotator import (
    MediapipeCompositeAnnotator,
)
from skellytracker.old.mediapipe_tracker.composite.mediapipe_composite_detector import MediapipeCompositeDetector
from skellytracker.old.mediapipe_tracker.composite.mediapipe_composite_tracker_config import \
    MediapipeCompositeTrackerConfig

logger = logging.getLogger(__name__)


class MediapipeCompositeRecorder(BaseRecorder):
    pass


class MediapipeCompositeTracker(BaseTracker):
    """
    Full-body holistic tracker combining pose, hand, and face detection.

    Recreates the behavior of the legacy MediaPipe Holistic pipeline by:
    1. Running pose detection on the full image
    2. Cropping hand/face regions guided by pose landmarks
    3. Running hand/face detection on those crops
    4. Merging results into a unified observation with fused body landmarks

    Usage:
        tracker = MediapipeCompositeTracker.create()
        observation = tracker.process_image(frame_number=0, image=rgb_image)
        annotated = tracker.annotate_image(image=rgb_image, observation=observation)
    """

    config: MediapipeCompositeTrackerConfig
    detector: MediapipeCompositeDetector
    annotator: MediapipeCompositeAnnotator
    recorder: MediapipeCompositeRecorder | None = None

    @classmethod
    def create(cls, config: MediapipeCompositeTrackerConfig | None = None) -> "MediapipeCompositeTracker":
        if config is None:
            config = MediapipeCompositeTrackerConfig()
        return cls(
            config=config,
            detector=MediapipeCompositeDetector.create(config=config.detector_config),
            annotator=MediapipeCompositeAnnotator.create(config=config.annotator_config),
            recorder=MediapipeCompositeRecorder(),
        )


if __name__ == "__main__":
    MediapipeCompositeTracker.create().demo()
