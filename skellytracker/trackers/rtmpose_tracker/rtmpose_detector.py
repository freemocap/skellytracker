from dataclasses import dataclass

import numpy as np
import onnxruntime
from numpy.typing import NDArray
from rtmlib import Wholebody

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetectorConfig, BaseDetector
from skellytracker.trackers.rtmpose_tracker.rtmpose_observation import RTMPoseObservation


class RTMPoseDetectorConfig(BaseDetectorConfig):
    confidence_threshold: float = 0.5
    mode: str = "performance"
    backend: str = "onnxruntime"
    device: str = "cuda"

@dataclass
class RTMPoseDetector(BaseDetector):
    config: RTMPoseDetectorConfig
    detector: Wholebody

    @classmethod
    def create(cls, config: RTMPoseDetectorConfig | None = None) -> "RTMPoseDetector":
        config = config or RTMPoseDetectorConfig()
        if config.device == "cuda":
            onnxruntime.preload_dlls()
        detector = Wholebody(
            to_openpose=False,
            mode=config.mode,
            backend=config.backend,
            device=config.device,
        )
        return cls(config=config, detector=detector)

    def detect(self, frame_number: int, image: NDArray[np.uint8]) -> RTMPoseObservation:
        # rtmlib's type stubs are incorrect — keypoints is float64 at runtime, scores is float32.
        keypoints: NDArray[np.float64]
        scores: NDArray[np.float32]
        keypoints, scores = self.detector(image)
        return RTMPoseObservation.from_detection_results(
            frame_number=frame_number,
            keypoints=keypoints,
            scores=scores,
            image_size=(int(image.shape[0]), int(image.shape[1])),
        )
