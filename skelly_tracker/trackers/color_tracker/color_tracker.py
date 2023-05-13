import logging

import cv2
import numpy as np

from skelly_tracker.trackers.tracker_base import TrackerBase

logger = logging.getLogger(__name__)


class ColorTracker(TrackerBase):

    def __init__(self,
                 lower_bound_color=[40, 40, 40],
                 upper_bound_color=[255, 255, 255]
                 ):
        self.lower_bound_color = np.array(lower_bound_color)
        self.upper_bound_color = np.array(upper_bound_color)
        self.tracking_data = None
        self.annotated_image = None

    def process_image(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self._mask_image = cv2.inRange(hsv_image, self.lower_bound_color, self.upper_bound_color)
        contours, _ = cv2.findContours(self._mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers = []
        for contour in contours:
            center, radius = cv2.minEnclosingCircle(contour)
            if radius > 1:
                centers.append(center)
                cv2.circle(image, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)

        self.tracking_data = {'centers': centers}
        self.annotated_image = image

    def get_output_data(self):
        return self.tracking_data

    def get_annotated_image(self):
        return self.annotated_image

    def demo(self, ):

        # Open webcam
        cap = cv2.VideoCapture(0)
        # Set initial exposure value
        exposure = -7
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame using the color tracker
            self.process_image(frame)

            # Get the annotated image and display it
            annotated_image = self.get_annotated_image()
            cv2.imshow('Color Tracker ("q" to quit)', annotated_image)

            # Exit if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Control exposure with up and down arrow keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ColorTracker().demo()
