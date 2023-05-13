import cv2
import cv2.aruco as aruco
from tracker_base import TrackerBase
from charuco_board_definition import CharucoBoardDefinition

class CharucoTracker(TrackerBase):

    def __init__(self, board_definition, aruco_params):
        self.board_definition = board_definition
        self.aruco_params = aruco_params

        self.tracking_data = None
        self.annotated_image = None

    def process_image(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(
            gray_image, self.board_definition.aruco_marker_dict, parameters=self.aruco_params
        )

        if ids is not None:
            _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray_image, self.board_definition.charuco_board
            )

            if charuco_corners is not None:
                self.tracking_data = {
                    "corners": charuco_corners,
                    "ids": charuco_ids,
                }
                self.annotated_image = aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)
            else:
                self.annotated_image = image.copy()
        else:
            self.annotated_image = image.copy()

    def get_output_data(self):
        return self.tracking_data

    def get_annotated_image(self):
        return self.annotated_image