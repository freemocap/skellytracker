import logging
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from skellytracker.trackers.rtmpose_tracker._skeleton_viz import draw_skeleton

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseImageAnnotatorConfig, BaseImageAnnotator
from skellytracker.trackers.rtmpose_tracker.rtmpose_observation import RTMPoseObservation
from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseImageAnnotatorConfig

logger = logging.getLogger(__name__)

# Bbox debug colors (BGR for OpenCV).
YOLOX_BBOX_COLOR = (0, 255, 0)       # green = YOLOX detection
TRACKING_BBOX_COLOR = (0, 165, 255)  # orange = tracking-predicted


@dataclass
class RTMPoseImageAnnotatorConfig(BaseImageAnnotatorConfig):
    confidence_threshold: float = 2.0
    # When True, draws the person bounding box on annotated images.
    # Green = from YOLOX detector, orange = from tracking prediction.
    draw_debug_bbox: bool = False


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

    # ------------------------------------------------------------------
    # Debug bbox drawing
    # ------------------------------------------------------------------

    @staticmethod
    def draw_bbox_on_image(
        image: NDArray[np.uint8],
        bbox: NDArray,
        *,
        from_detector: bool = True,
        label: str | None = None,
        thickness: int = 2,
    ) -> NDArray[np.uint8]:
        """Draw a person bounding box rectangle on *image* (mutates in-place).

        Args:
            image: BGR uint8 image (mutated in-place).
            bbox: ``(4,)`` xyxy array in image pixel coords.
            from_detector: True → green (YOLOX), False → orange (tracking).
            label: Optional text label drawn above the bbox.
            thickness: Line thickness in pixels.
        """
        if bbox is None or len(bbox) == 0:
            return image
        # Handle (N,4) — take first row.
        if bbox.ndim == 2:
            bbox = bbox[0]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = YOLOX_BBOX_COLOR if from_detector else TRACKING_BBOX_COLOR
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        if label is not None:
            cv2.putText(
                image, label, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

        return image
