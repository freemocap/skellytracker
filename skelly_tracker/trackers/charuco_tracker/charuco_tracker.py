import cv2

from skelly_tracker.trackers.base_tracker.base_tracker import BaseTracker


class CharucoTracker(BaseTracker):
    def __init__(self,
                 squaresX: int,
                 squaresY: int,
                 dictionary: cv2.aruco_Dictionary,
                 squareLength: float = 1,
                 markerLength: float = .8,
                 ):
        super().__init__(tracked_object_names=["charuco_corners", "charuco_ids"])
        self.board = cv2.aruco.CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, dictionary)

    def process_image(self, image, **kwargs):
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
                # Draw the Charuco corners
                cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

                # Update the tracking data
                self.tracking_data = {
                    "charuco_corners": charuco_corners,
                    "charuco_ids": charuco_ids
                }

        self.annotated_image = image

        return {
            "tracking_data": self.tracking_data,
            "annotated_image": self.annotated_image,
            "raw_image": image,
        }


if __name__ == "__main__":
    CharucoTracker(squaresX=7,
                        squaresY=5,
                        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)).demo()
