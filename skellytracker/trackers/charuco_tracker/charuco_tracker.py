from typing import Dict, List

import cv2
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject


class CharucoTracker(BaseTracker):
    def __init__(self,
                 tracked_object_names: List[str],
                 squares_x: int,
                 squares_y: int,
                 dictionary: cv2.aruco_Dictionary,
                 squareLength: float = 1,
                 markerLength: float = .8,
                 ):
        super().__init__(recorder=None, tracked_object_names=tracked_object_names)
        self.board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, squareLength, markerLength, dictionary)

    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray_image, self.board.dictionary)

        # If any markers were found
        if len(corners) > 0:
            # Refine the detected markers
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray_image,
                                                                                    self.board)

            # If any Charuco corners were found
            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                # Clear previous tracked objects
                self.tracked_objects.clear()

                # Create a TrackedObject for each corner
                for i, corner in enumerate(charuco_corners):
                    object_id = str(i)
                    self.tracked_objects[object_id] = TrackedObject(object_id=object_id)
                    self.tracked_objects[object_id].pixel_x = corner[0][0]
                    self.tracked_objects[object_id].pixel_y = corner[0][1]

        self.annotated_image = self.annotate_image(image=image,
                                                   tracked_objects=self.tracked_objects)

        return self.tracked_objects

    def annotate_image(self, image: np.ndarray, tracked_objects: Dict[str, TrackedObject], **kwargs) -> np.ndarray:
        # Copy the original image for annotation
        annotated_image = image.copy()

        # Draw a marker for each tracked corner
        for tracked_object in tracked_objects.values():
            if tracked_object.pixel_x is not None and tracked_object.pixel_y is not None:
                cv2.drawMarker(annotated_image,
                               (int(tracked_object.pixel_x), int(tracked_object.pixel_y)),
                               (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        return annotated_image


if __name__ == "__main__":
    charuco_squares_x_in = 7
    charuco_squares_y_in = 5
    number_of_charuco_markers = charuco_squares_x_in - 1 * charuco_squares_y_in - 1
    charuco_ids = [str(index) for index in range(number_of_charuco_markers)]

    CharucoTracker(tracked_object_names=charuco_ids,
                   squares_x=charuco_squares_x_in,
                   squares_y=charuco_squares_y_in,
                   dictionary=cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
                   ).demo()
