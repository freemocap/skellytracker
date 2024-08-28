from typing import Dict, List

import cv2
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.charuco_tracker.charuco_recorder import CharucoRecorder


default_aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

class CharucoTracker(BaseTracker):
    def __init__(
        self,
        tracked_object_names: List[str],
        squares_x: int,
        squares_y: int,
        dictionary: cv2.aruco.Dictionary = default_aruco_dictionary,
        square_length: float = 1,
        marker_length: float = 0.8,
    ):
        super().__init__(
            recorder=CharucoRecorder(), tracked_object_names=tracked_object_names
        )
        self.board = cv2.aruco.CharucoBoard(
            size=(squares_x, squares_y),
            squareLength=square_length,
            markerLength=marker_length,
            dictionary=dictionary,
        )

        # Following most recent charuco detection documentation: https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)

        self.tracked_object_names = tracked_object_names
        self.dictionary = dictionary

    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, _marker_corners, _marker_ids = (
            self.charuco_detector.detectBoard(gray_image)
        )

        self.reinitialize_tracked_objects()

        # If any Charuco corners were found
        if (
            charuco_corners is not None
            and charuco_ids is not None
            and len(charuco_corners) > 3
        ):
            # Create a TrackedObject for each corner
            for id, corner in zip(charuco_ids, charuco_corners):
                object_id = str(id).strip("[]")
                self.tracked_objects[object_id] = TrackedObject(object_id=object_id)
                self.tracked_objects[object_id].pixel_x = corner[0][0]
                self.tracked_objects[object_id].pixel_y = corner[0][1]

        self.annotated_image = self.annotate_image(
            image=image, tracked_objects=self.tracked_objects
        )

        return self.tracked_objects

    def annotate_image(
        self, image: np.ndarray, tracked_objects: Dict[str, TrackedObject], **kwargs
    ) -> np.ndarray:
        # Copy the original image for annotation
        annotated_image = image.copy()

        # Draw a marker for each tracked corner
        for tracked_object in tracked_objects.values():
            if (
                tracked_object.pixel_x is not None
                and tracked_object.pixel_y is not None
            ):
                cv2.drawMarker(
                    annotated_image,
                    (int(tracked_object.pixel_x), int(tracked_object.pixel_y)),
                    (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=30,
                    thickness=2,
                )
                cv2.putText(
                    annotated_image,
                    tracked_object.object_id,
                    (int(tracked_object.pixel_x), int(tracked_object.pixel_y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

        return annotated_image

    def reinitialize_tracked_objects(self) -> None:
        """
        Reinitialize tracked objects to clear previous frames data

        Unlike self.tracked_objects.clear(), this will ensure every tracked object has a value for each frame, even if its empty
        """
        for name in self.tracked_object_names:
            self.tracked_objects[name] = TrackedObject(object_id=name)


if __name__ == "__main__":
    charuco_squares_x_in = 7
    charuco_squares_y_in = 5
    number_of_charuco_markers = (charuco_squares_x_in - 1) * (charuco_squares_y_in - 1)
    charuco_ids = [str(index) for index in range(number_of_charuco_markers)]

    CharucoTracker(
        tracked_object_names=charuco_ids,
        squares_x=charuco_squares_x_in,
        squares_y=charuco_squares_y_in,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
    ).demo()
