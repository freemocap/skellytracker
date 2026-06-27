import numpy as np
from easy_ViTPose.vit_utils.visualization import draw_points_and_skeleton, joints_dict

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseImageAnnotatorConfig, BaseImageAnnotator
from skellytracker.trackers.vitpose_tracker.vitpose_observation import VITPoseObservation


class VITPoseAnnotator(BaseImageAnnotator):
    config: BaseImageAnnotatorConfig
    observations: list[VITPoseObservation]

    @classmethod
    def create(cls, config: BaseImageAnnotatorConfig | None = None):
        if config is None:
            config = BaseImageAnnotatorConfig()
        return cls(config=config, observations=[])

    def annotate_image(self, image: np.ndarray, observation: VITPoseObservation) -> np.ndarray:

        annotated_image = draw_points_and_skeleton(
            image=image,
            skeleton=joints_dict()['wholebody']['skeleton'],
            points=observation.keypoints,
            confidence_threshold=0.5) #should find a way to connect this to the model confidence threshold
        
        return annotated_image
