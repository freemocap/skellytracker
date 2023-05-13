import logging
import cv2
import numpy as np

from skelly_tracker.trackers.base_tracker.base_tracker import BaseTracker

UPPER_BOUND_COLOR = [255, 255, 255]

LOWER_BOUND_COLOR = [40, 40, 40]

logger = logging.getLogger(__name__)

class ColorTracker(BaseTracker):
    lower_bound_color: np.ndarray = np.array(LOWER_BOUND_COLOR)
    upper_bound_color: np.ndarray = np.array(UPPER_BOUND_COLOR)
    mask_image: np.ndarray = None

    def __init__(self):
        super().__init__(tracked_object_names=["color"],
                         process_image_function=self.process_image)

    def process_image(self, image, **kwargs):
        raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.mask_image = cv2.inRange(raw_image, self.lower_bound_color, self.upper_bound_color)
        contours, _ = cv2.findContours(self.mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            center, radius = cv2.minEnclosingCircle(contour)
            if radius > 1:
                self.tracked_objects["color"]["pixel_locations"].append(center)
                self.tracked_objects["color"]["bounding_boxes"].append((int(center[0] - radius), int(center[1] - radius), int(2 * radius), int(2 * radius)))
                cv2.circle(image, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)

        self.tracking_data[len(self.tracking_data)] = self.tracked_objects
        self.annotated_image = image
        return {
            "tracking_data": self.tracking_data,
            "annotated_image": self.annotated_image,
            "raw_image": raw_image,
        }


if __name__ == "__main__":
    ColorTracker().demo()
