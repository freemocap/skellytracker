import cv2
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseImageAnnotator,
    BaseImageAnnotatorConfig,
    BaseObservation,
)
from skellytracker.trackers.mediapipe_tracker.body.mediapipe_pose_annotator import MediapipePoseAnnotator, \
    MediapipePoseAnnotatorConfig
from skellytracker.trackers.mediapipe_tracker.composite.mediapipe_composite_observation import (
    MediapipeCompositeObservation,
    ROIBox,
)
from skellytracker.trackers.mediapipe_tracker.face.mediapipe_face_annotator import MediapipeFaceAnnotator, \
    MediapipeFaceAnnotatorConfig
from skellytracker.trackers.mediapipe_tracker.hands.mediapipe_hand_annotator import MediapipeHandAnnotator, \
    MediapipeHandAnnotatorConfig


class MediapipeCompositeAnnotatorConfig(BaseImageAnnotatorConfig):
    show_overlay: bool = True
    pose_config: MediapipePoseAnnotatorConfig = MediapipePoseAnnotatorConfig()
    hand_config: MediapipeHandAnnotatorConfig = MediapipeHandAnnotatorConfig()
    face_config: MediapipeFaceAnnotatorConfig = MediapipeFaceAnnotatorConfig()

    # ROI box drawing
    draw_roi_boxes: bool = True
    hand_roi_color: tuple[int, int, int] = (0, 255, 0)
    face_roi_color: tuple[int, int, int] = (255, 255, 0)
    roi_thickness: int = 2


class MediapipeCompositeAnnotator(BaseImageAnnotator):
    config: MediapipeCompositeAnnotatorConfig
    observations: list[MediapipeCompositeObservation]

    pose_annotator: MediapipePoseAnnotator
    hand_annotator: MediapipeHandAnnotator
    face_annotator: MediapipeFaceAnnotator

    @classmethod
    def create(cls, config: MediapipeCompositeAnnotatorConfig) -> "MediapipeCompositeAnnotator":
        return cls(
            config=config,
            observations=[],
            pose_annotator=MediapipePoseAnnotator.create(config=config.pose_config),
            hand_annotator=MediapipeHandAnnotator.create(config=config.hand_config),
            face_annotator=MediapipeFaceAnnotator.create(config=config.face_config),
        )

    def annotate_image(self, image: np.ndarray, observation: BaseObservation) -> np.ndarray:
        if not isinstance(observation, MediapipeCompositeObservation):
            raise TypeError(f"Expected MediapipeCompositeObservation, got {type(observation)}")

        annotated = image.copy()

        # Draw pose skeleton
        if observation.pose is not None and observation.pose.has_detection:
            annotated = self.pose_annotator.annotate_image(image=annotated, observation=observation.pose)

        # Draw hands
        if observation.hands is not None and observation.hands.has_detection:
            annotated = self.hand_annotator.annotate_image(image=annotated, observation=observation.hands)

        # Draw face
        if observation.face is not None and observation.face.has_detection:
            annotated = self.face_annotator.annotate_image(image=annotated, observation=observation.face)

        # Draw ROI boxes
        if self.config.draw_roi_boxes:
            self._draw_roi_box(image=annotated, roi=observation.left_hand_roi, color=self.config.hand_roi_color, label="L Hand")
            self._draw_roi_box(image=annotated, roi=observation.right_hand_roi, color=self.config.hand_roi_color, label="R Hand")
            self._draw_roi_box(image=annotated, roi=observation.face_roi, color=self.config.face_roi_color, label="Face")

        return annotated

    def _draw_roi_box(
        self,
        image: np.ndarray,
        roi: ROIBox | None,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        if roi is None:
            return
        cv2.rectangle(
            image,
            (roi.x, roi.y),
            (roi.x + roi.width, roi.y + roi.height),
            color=color,
            thickness=self.config.roi_thickness,
        )
        self.draw_doubled_text(
            image=image,
            text=label,
            x=roi.x,
            y=roi.y - 5,
            font_scale=0.4,
            color=color,
            thickness=1,
        )
