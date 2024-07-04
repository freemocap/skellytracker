import logging
from typing import Dict

import cv2
import numpy as np
from pydantic import BaseModel

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.bright_point_tracker.brightest_point_recorder import (
    BrightestPointRecorder,
)

UPPER_BOUND_COLOR = [255, 255, 255]

LOWER_BOUND_COLOR = [40, 40, 40]

logger = logging.getLogger(__name__)


class BrightPatch(BaseModel):
    area: float
    centroid_x: int
    centroid_y: int


class BrightestPointTracker(BaseTracker):
    def __init__(self, num_points: int = 1, luminance_threshold: int = 200):
        super().__init__(
            tracked_object_names=[f"brightest_point_{i}" for i in range(num_points)],
            recorder=BrightestPointRecorder(),
        )

        self.num_points = num_points
        self.luminance_threshold = luminance_threshold

    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to get only bright regions
        _, thresholded_image = cv2.threshold(
            gray_image, self.luminance_threshold, 255, cv2.THRESH_BINARY
        )

        # Find contours of the bright regions
        bright_patches, _ = cv2.findContours(
            thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Process each bright patch separately
        patch_list = []
        for patch in bright_patches:
            patch_moments = cv2.moments(patch)
            if patch_moments["m00"] != 0:  # Avoid division by zero
                centroid_x = int(patch_moments["m10"] / patch_moments["m00"])
                centroid_y = int(patch_moments["m01"] / patch_moments["m00"])

                patch_list.append(
                    BrightPatch(
                        area=cv2.contourArea(patch),
                        centroid_x=centroid_x,
                        centroid_y=centroid_y,
                    )
                )

        largest_patches = sorted(
            patch_list, key=lambda patch: patch.area, reverse=True
        )[: self.num_points]

        for i, patch in enumerate(largest_patches):
            self.tracked_objects[f"brightest_point_{i}"].pixel_x = patch.centroid_x
            self.tracked_objects[f"brightest_point_{i}"].pixel_y = patch.centroid_y
            self.tracked_objects[f"brightest_point_{i}"].extra[
                "thresholdedimage"
            ] = thresholded_image

        for i in range(len(largest_patches), self.num_points):
            self.tracked_objects[f"brightest_point_{i}"].pixel_x = (
                None  # TODO: Is this the right value for missing data?
            )
            self.tracked_objects[f"brightest_point_{i}"].pixel_y = None

        self.annotated_image = self.annotate_image(
            image=image, tracked_objects=self.tracked_objects
        )

        return self.tracked_objects

    def annotate_image(
        self, image: np.ndarray, tracked_objects: Dict[str, TrackedObject], **kwargs
    ) -> np.ndarray:
        annotated_image = image.copy()

        for key, tracked_object in tracked_objects.items():
            if (
                "brightest_point" in key
                and tracked_object.pixel_x is not None
                and tracked_object.pixel_y is not None
            ):
                cv2.drawMarker(
                    img=annotated_image,
                    position=(int(tracked_object.pixel_x), int(tracked_object.pixel_y)),
                    color=(0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=2,
                )

        return annotated_image


if __name__ == "__main__":
    BrightestPointTracker(num_points=2).demo()
