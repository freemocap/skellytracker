import cv2
import numpy as np
from skellytracker.trackers.base_tracker.base_tracker import BaseImageAnnotator, BaseImageAnnotatorConfig
from skellytracker.trackers.bright_point_tracker.brightest_point_observation import BrightestPointObservation


class BrightestPointAnnotatorConfig(BaseImageAnnotatorConfig):
    show_tracks: int | None = 15
    color: tuple[int, int, int] = (0, 0, 255)
    marker_size: int = 20
    marker_type: int = cv2.MARKER_CROSS
    thickness: int = 2

class BrightestPointImageAnnotator(BaseImageAnnotator):
    config: BrightestPointAnnotatorConfig
    observations: list[BrightestPointObservation]

    @classmethod
    def create(cls, config: BrightestPointAnnotatorConfig):
        return cls(config=config, observations=[])
    
    def annotate_image(self, image: np.ndarray, latest_observation: BrightestPointObservation) -> np.ndarray:
        annotated_image = image.copy()

        for bright_patch in latest_observation.bright_patches:
            if (
                bright_patch is not None
                and bright_patch.centroid_x is not None
                and bright_patch.centroid_y is not None
            ):
                cv2.drawMarker(
                    img=annotated_image,
                    position=(int(bright_patch.centroid_x), int(bright_patch.centroid_y)),
                    color=self.config.color,
                    markerType=self.config.marker_type,
                    markerSize=self.config.marker_size,
                    thickness=self.config.thickness,
                )

        return annotated_image