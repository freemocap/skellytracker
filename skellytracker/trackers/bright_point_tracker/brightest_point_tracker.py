import logging
from typing import Dict

import cv2
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.bright_point_tracker.brightest_point_recorder import BrightestPointRecorder

UPPER_BOUND_COLOR = [255, 255, 255]

LOWER_BOUND_COLOR = [40, 40, 40]

logger = logging.getLogger(__name__)


class BrightestPointTracker(BaseTracker):
    luminance_threshold: int = 200

    def __init__(self):
        super().__init__(tracked_object_names=["brightest_point"], recorder=BrightestPointRecorder())

    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to get only bright regions
        _, thresholded_image = cv2.threshold(gray_image, self.luminance_threshold, 255, cv2.THRESH_BINARY)

        # Find contours of the bright regions
        bright_patches, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each bright patch separately
        largest_area = 0
        largest_patch_centroid = None
        for patch in bright_patches:
            # Calculate the centroid of the bright patch
            patch_moments = cv2.moments(patch)
            if patch_moments["m00"] != 0:  # Avoid division by zero
                centroid_x = int(patch_moments["m10"] / patch_moments["m00"])
                centroid_y = int(patch_moments["m01"] / patch_moments["m00"])

                # Keep track of the largest patch and its centroid
                area = cv2.contourArea(patch)
                if area > largest_area:
                    largest_area = area
                    largest_patch_centroid = (centroid_x, centroid_y)

        # If a largest patch was found, update the tracked object
        if largest_patch_centroid is not None:
            self.tracked_objects["brightest_point"].pixel_x = largest_patch_centroid[0]
            self.tracked_objects["brightest_point"].pixel_y = largest_patch_centroid[1]
            self.tracked_objects["brightest_point"].extra["thresholded_image"] = thresholded_image

        self.raw_image = image.copy()

        self.annotated_image = self.annotate_image(image=image,
                                                   tracked_objects=self.tracked_objects)

        return self.tracked_objects

    def annotate_image(self, image: np.ndarray, tracked_objects: Dict[str, TrackedObject], **kwargs) -> np.ndarray:
        # Copy the original image for annotation
        annotated_image = image.copy()

        # Draw a red 'X' over the largest bright patch
        if tracked_objects["brightest_point"].pixel_x is not None and tracked_objects[
            "brightest_point"].pixel_y is not None:
            cv2.drawMarker(annotated_image,
                           (tracked_objects["brightest_point"].pixel_x, tracked_objects["brightest_point"].pixel_y),
                           (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        return annotated_image


if __name__ == "__main__":
    BrightestPointTracker().demo()
