from skelly_tracker.trackers.base_tracker.base_tracker import BaseTracker
from skelly_tracker.trackers.base_tracker.tracking_data_model import TrackedObject, FrameData
import cv2
import numpy as np

class BrightestPointTracker(BaseTracker):

    def process_image(self, image: np.ndarray):
        self.raw_image = image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray_image)

        # Define a bounding box around the brightest pixel
        x, y = maxLoc
        box_size = 20  # Define the size of the box
        bounding_box = ((x - box_size, y - box_size), (x + box_size, y + box_size))

        # Create an annotated image
        self.annotated_image = image.copy()
        cv2.drawMarker(self.annotated_image, maxLoc, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.rectangle(self.annotated_image, bounding_box[0], bounding_box[1], (0, 255, 0), 2)

        # Create a TrackedObject and add it to the tracking data
        tracked_object = TrackedObject(object_id="brightest_pixel")
        tracked_object.add_data(pixel_location=maxLoc, bounding_box=bounding_box)
        frame_data = FrameData()
        frame_data.add_tracked_object(object_id="brightest_pixel", pixel_location=maxLoc, bounding_box=bounding_box)
        self.tracking_data.add_frame_data(frame_number=0, frame_data=frame_data)  # Assuming frame_number=0 for simplicity

    def get_tracking_data(self):
        return self.tracking_data

    def get_annotated_image(self):
        return self.annotated_image
if __name__ == "__main__":
    BrightestPointTracker().demo()