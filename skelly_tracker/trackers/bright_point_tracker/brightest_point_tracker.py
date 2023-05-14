import logging
import cv2
import numpy as np

from skelly_tracker.trackers.base_tracker.base_tracker import BaseTracker

UPPER_BOUND_COLOR = [255, 255, 255]

LOWER_BOUND_COLOR = [40, 40, 40]

logger = logging.getLogger(__name__)

class BrightestPointTracker(BaseTracker):
    luminance_threshold: int = 200


    def __init__(self):
        super().__init__(tracked_object_names=["brightest_points"],
                         process_image_function=self.process_image)

    def process_image(self, image,**kwargs):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to get only bright regions
        _, thresholded = cv2.threshold(gray_image, self.luminance_threshold, 255, cv2.THRESH_BINARY)

        # Find the coordinates of the brightest point
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(thresholded)

        # Draw a circle around the brightest point
        cv2.circle(image, max_loc, 20, (0, 255, 0), 2)

        # Update the tracking data
        self.tracking_data = {"brightest_points": {"pixel": {"x": max_loc[0], "y": max_loc[1]}}}
        self.annotated_image = image

        return {
            "tracking_data": self.tracking_data,
            "annotated_image": self.annotated_image,
            "raw_image": image,
        }


if __name__ == "__main__":
    BrightestPointTracker().demo()
