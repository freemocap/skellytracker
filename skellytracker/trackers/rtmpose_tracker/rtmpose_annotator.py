from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skellytracker.trackers.rtmpose_tracker._skeleton_viz import draw_skeleton

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseImageAnnotatorConfig, BaseImageAnnotator
from skellytracker.trackers.rtmpose_tracker.rtmpose_observation import RTMPoseObservation
from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseImageAnnotatorConfig


@dataclass
class RTMPoseImageAnnotatorConfig(BaseImageAnnotatorConfig):
    confidence_threshold: float = 2.0


@dataclass
class RTMPoseImageAnnotator(BaseImageAnnotator):
    config: RTMPoseImageAnnotatorConfig
    observations: list[RTMPoseObservation]

    @classmethod
    def create(cls, config: RTMPoseImageAnnotatorConfig | None = None) -> "RTMPoseImageAnnotator":
        if config is None:
            config = RTMPoseImageAnnotatorConfig()
        return cls(config=config, observations=[])

    def annotate_image(
        self,
        image: NDArray[np.uint8],
        observation: RTMPoseObservation | None = None,
    ) -> NDArray[np.uint8]:
        if observation is None or observation.keypoints.shape[0] == 0:
            return image
        annotated_image = draw_skeleton(
            img=image,
            keypoints=observation.keypoints,
            # RTMPose confidence scores are in arbitrary units based on heatmap
            # peak height. A threshold of 2.0 filters out weak detections better
            # than the default 0.5.
            kpt_thr=self.config.confidence_threshold,
            scores=observation.scores,
        )
        return annotated_image

    def annotate_image_from_keypoints_and_scores(
        self,
        image: NDArray[np.uint8],
        keypoints: NDArray[np.float32],
        scores: NDArray[np.float32],
    ) -> NDArray[np.uint8]:
        annotated_image = draw_skeleton(img=image, keypoints=keypoints, scores=scores)
        return annotated_image
