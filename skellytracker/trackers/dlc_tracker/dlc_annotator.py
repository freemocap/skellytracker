import cv2
import numpy as np
from numpydantic import NDArray, Shape


from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseImageAnnotator, BaseImageAnnotatorConfig
from skellytracker.trackers.dlc_tracker.dlc_observation import DeepLabCutObservation


class DeepLabCutAnnotatorConfig(BaseImageAnnotatorConfig):
    show_tracks: int | None = 15
    show_overlay: bool = True
    marker_type: int = cv2.MARKER_DIAMOND
    marker_size: int = 10
    marker_thickness: int = 2
    marker_color: tuple[int, int, int] = (0, 0, 255)

    text_color: tuple[int, int, int] = (215, 115, 40)
    text_size: float = .5
    text_thickness: int = 2
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX


class DeepLabCutImageAnnotator(BaseImageAnnotator):
    config: DeepLabCutAnnotatorConfig
    observations: list[DeepLabCutObservation]

    @classmethod
    def create(cls, config: DeepLabCutAnnotatorConfig):
        return cls(config=config, observations=[])
    
    def annotate_image(
            self,
            image: NDArray[Shape["* width, * height, 1-4 channels"], np.uint8],
            latest_observation: DeepLabCutObservation | None = None,
    ) -> np.ndarray:
        if latest_observation is None:
            return image.copy()
        # Copy the original image for annotation
        annotated_image = image.copy()

        for marker in range(latest_observation.pose_points.shape[0]):
            point = latest_observation.pose_points[marker, :2]
            cv2.drawMarker(
                img=annotated_image,
                position=(int(point[0]), int(point[1])),
                color=self.config.marker_color,
                markerType=self.config.marker_type,
                markerSize=self.config.marker_size,
                thickness=self.config.marker_thickness,
            )

        return annotated_image
