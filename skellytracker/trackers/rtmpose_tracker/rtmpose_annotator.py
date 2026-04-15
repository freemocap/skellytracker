import numpy as np
from rtmlib import draw_skeleton

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseImageAnnotatorConfig, BaseImageAnnotator
from skellytracker.trackers.rtmpose_tracker.rtmpose_observation import RTMPoseObservation


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
                                        kpt_thr=2, # RTMPose confidence scores are in arbitrary units based on the height of the heatmap detection of each keypoint. Default cutoff (0.5) seems too low. Cutting off below 2.0 seems to work ok.
                                        scores=observation.scores)
        return annotated_image
    
    def annotate_image_from_keypoints_and_scores(
            self,
            image: np.ndarray,
            keypoints: np.ndarray,
            scores: np.ndarray
    ):
        annotated_image = draw_skeleton(img=image, 
                                        keypoints=keypoints,
                                        scores=scores)
        return annotated_image
