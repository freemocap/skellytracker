from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseTracker, BaseTrackerConfig, BaseRecorder
from skellytracker.trackers.rtmpose_tracker.rtmpose_annotator import RTMPoseImageAnnotator, BaseImageAnnotatorConfig
from skellytracker.trackers.rtmpose_tracker.rtmpose_detector import RTMPoseDetector, RTMPoseDetectorConfig


class RTMPoseTrackerConfig(BaseTrackerConfig):
    detector_config: RTMPoseDetectorConfig = Field(default_factory=RTMPoseDetectorConfig)
    annotator_config: BaseImageAnnotatorConfig | None = None

class RTMPoseRecorder(BaseRecorder):
    pass

class RTMPoseTracker(BaseTracker):
    config: RTMPoseTrackerConfig
    detector: RTMPoseDetector
    annotator: RTMPoseImageAnnotator | None = None
    recorder: RTMPoseRecorder | None = None

    @classmethod
    def create(cls, config: RTMPoseTrackerConfig | None = None):
        if config is None:
            config = RTMPoseTrackerConfig()
        detector = RTMPoseDetector.create(config.detector_config)

        return cls(
            config = config,
            detector = detector,
            annotator = RTMPoseImageAnnotator.create(config.annotator_config),
            recorder = RTMPoseRecorder(),
        )

if __name__ == "__main__":
    import onnxruntime as ort

    ort.preload_dlls()
    print(f"ort.get_available_providers() -> {ort.get_available_providers()}")
    RTMPoseTracker.create().demo()
