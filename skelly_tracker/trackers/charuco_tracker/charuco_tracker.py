from typing import Any, Optional, List

import cv2 as cv

from skelly_tracker.trackers.base_tracker.base_tracker import BaseTracker
from skelly_tracker.trackers.charuco_tracker.charuco_board_definition import CharucoBoardDefinition

class CharucoTracker(BaseTracker):
    def __init__(self, board_definition: CharucoBoardDefinition = CharucoBoardDefinition()):
        self.board_definition = board_definition
        self.detector = cv.aruco.CharucoDetector_create(self.board_definition.charuco_board)
        super().__init__(tracked_object_names=self.charuco_ids)

    def process_image(self, image, **kwargs):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray_image)
        if ids is not None:
            _, charuco_corners, charuco_ids = self.detector.interpolateCornersCharuco(corners, ids, gray_image, self.board_definition.charuco_board)
            if charuco_corners is not None:
                for i, id in enumerate(charuco_ids.flatten()):
                    self.tracked_objects[str(id)].add_data(pixel_location=charuco_corners[i].ravel())
                self.annotated_image = cv.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)
            else:
                self.annotated_image = image.copy()
        else:
            self.annotated_image = image.copy()

if __name__ == "__main__":
    CharucoTracker().demo()