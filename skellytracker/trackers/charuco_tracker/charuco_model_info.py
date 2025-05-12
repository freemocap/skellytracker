from skellytracker.trackers.base_tracker.model_info import ModelInfo
from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams

import cv2

class CharucoModelInfo(ModelInfo):
    name = "charuco"
    tracker_name = 'CharucoTracker'

class CharucoTrackingParams(BaseTrackingParams):
    charuco_squares_x_in: int = 7
    charuco_squares_y_in: int = 5
    charuco_dict_id: int = cv2.aruco.DICT_4X4_250


