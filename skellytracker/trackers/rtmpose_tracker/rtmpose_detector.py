import numpy as np
from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetectorConfig, BaseDetector
from skellytracker.trackers.rtmpose_tracker.rtmpose_observation import RTMPoseObservation
from rtmlib import Wholebody

class RTMPoseDetectorConfig(BaseDetectorConfig):
    confidence_threshold: float = 0.5
    mode = 'balanced'
    backend = 'onnxruntime'
    device = 'cpu'

class RTMPoseDetector(BaseDetector):
    config: RTMPoseDetectorConfig
    detector: Wholebody

    @classmethod
    def create(cls, config: RTMPoseDetectorConfig | None = None):
        config = config or RTMPoseDetectorConfig()
        detector = Wholebody(
            to_openpose=False, #last time I set to true this failed
            mode=config.mode,
            backend=config.backend,
            device=config.device
        )  
        return cls(
            config=config, 
            detector=detector)

    def detect(self, frame_number: int, image: np.ndarray) -> RTMPoseObservation:
        keypoints, scores = self.detector(image)
        return RTMPoseObservation.from_detection_results(
            frame_number=frame_number,
            results=results,
            image_size=(int(image.shape[0]), int(image.shape[1])),
        )