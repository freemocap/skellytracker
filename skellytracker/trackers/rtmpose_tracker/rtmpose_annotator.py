from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseImageAnnotatorConfig, BaseImageAnnotator
from skellytracker.trackers.rtmpose_tracker.rtmpose_observation import RTMPoseObservation
import numpy as np
from rtmlib import draw_skeleton

class RTMPoseImageAnnotator(BaseImageAnnotator):
    config: BaseImageAnnotatorConfig
    observations: list[RTMPoseObservation]

    @classmethod
    def create(cls, config: BaseImageAnnotatorConfig | None = None) -> "RTMPoseImageAnnotator":
        if config is None:
            config = BaseImageAnnotatorConfig()
        return cls(config=config, observations=[])
    
    def annotate_image(
            self,
            image: np.ndarray,
            observation: RTMPoseObservation | None = None,
    ):
        annotated_image = draw_skeleton(img=image, 
                                        keypoints=observation.keypoints,
                                        scores=observation.scores)
        return annotated_image
