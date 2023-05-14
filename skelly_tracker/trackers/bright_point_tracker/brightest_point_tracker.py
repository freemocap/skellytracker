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

    def process_image(self, image, **kwargs):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to get only bright regions
        _, thresholded = cv2.threshold(gray_image, self.luminance_threshold, 255, cv2.THRESH_BINARY)

        # Find contours of the bright regions
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # For storing bright points and corresponding centroids
        self.tracking_data = {"brightest_points": []}

        # Process each bright patch separately
        for contour in contours:
            # Calculate the centroid of the bright patch
            M = cv2.moments(contour)
            M = cv2.moments(contour)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw an 'X' on the centroid
                cv2.drawMarker(image, (cX, cY), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

                # Draw minimum enclosing circle around the bright patch
                center, radius = cv2.minEnclosingCircle(contour)
                cv2.circle(image, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)

                # Add to tracking data
                self.tracking_data["brightest_points"].append({"pixel": {"x": cX, "y": cY}})

        # Color all the pixels that exceed the threshold in Blue
        image[thresholded == 255] = [255, 0, 0]

        self.annotated_image = image

        return {
            "tracking_data": self.tracking_data,
            "annotated_image": self.annotated_image,
            "raw_image": image,
        }



if __name__ == "__main__":
    BrightestPointTracker().demo()
