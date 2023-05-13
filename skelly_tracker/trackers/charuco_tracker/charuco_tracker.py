from typing import Any, Optional, List

import cv2 as cv

from skelly_tracker.trackers.base_tracker.base_tracker import BaseTracker



class CharucoBoardDefinition:
    def __init__(self,
                 aruco_marker_dict_id=cv.aruco.DICT_4X4_250,
                 number_of_squares_width=7,
                 number_of_squares_height=5,
                 black_square_side_length=1,
                 aruco_marker_length_proportional=0.8
                 ):

        self.aruco_marker_dict = cv.aruco.getPredefinedDictionary(aruco_marker_dict_id)
        self.number_of_squares_width = number_of_squares_width
        self.number_of_squares_height = number_of_squares_height
        self.black_square_side_length = black_square_side_length
        self.aruco_marker_length_proportional = aruco_marker_length_proportional
        self.charuco_board = cv.aruco.CharucoBoard(
            size=[self.number_of_squares_width, self.number_of_squares_height],
            squareLength=self.black_square_side_length,
            markerLength=self.aruco_marker_length_proportional,
            dictionary=self.aruco_marker_dict,
        )
        self.number_of_charuco_corners = (self.number_of_squares_width - 1) * (self.number_of_squares_height - 1)

class CharucoTracker(BaseTracker):
    board_definition: Optional[CharucoBoardDefinition] = None
    def __init__(self, board_definition: CharucoBoardDefinition = None):
        if board_definition is None:
            self.board_definition = CharucoBoardDefinition()

        self.dictionary = self.board_definition.aruco_marker_dict.get_dictionary()
        self.parameters = cv.aruco.DetectorParameters_create()
        self.detector = cv.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.charuco_ids = list(str(i) for i in range(self.board_definition.number_of_charuco_corners))
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
