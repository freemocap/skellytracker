from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict
from numpydantic import NDArray, Shape

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation, TrackerTypeString, TrackedPoint2dArray, \
    TrackedPointIdString

AllCharucoCorners3DByIdInObjectCoordinates = NDArray[Shape["* charuco_id, 3 xyz"], np.float32]
AllArucoCorners3DByIdInObjectCoordinates = NDArray[Shape["* aruco_ids, 4 corners, 3 xyz"], np.float32]
DetectedCharucoCornerIds = NDArray[Shape["* charuco_id, ..."], int]
RawCharucoCornersImageCoordinates = NDArray[Shape["* charuco_id,1 dim,  2 pxpy"], float]
DetectedCharucoCornersImageCoordinates = NDArray[Shape["* charuco_id, 2 pxpy"], float]
DetectedCharucoCornersInObjectCoordinates = NDArray[Shape["* charuco_id, 3 xyz"], float]

DetectedCharucoCorners2DInFullArray = NDArray[Shape["* charuco_id, 2 pxpy"], float] # (i.e. a 2D array where the index corresponds to the charuco id, non-detected corners are set to np.nan)

ArucoMarkerCorners = NDArray[Shape["4 corners, 2 pxpy"], float]
DetectedArucoMarkerIds = NDArray[Shape["* aruco_id, ..."], int] # ID of the corresponding entry in the DetectedArucoMarkerCorners tuple
DetectedArucoMarkerCorners = Sequence[NDArray[Shape[" 4 corners, 2 pxpy"], float]]
CharucoBoardTranslationVector = NDArray[Shape["3 tx_ty_tz"], np.float32]
CharucoBoardRotationVector = NDArray[Shape["3 rx_ry_rz"], np.float32]

class AniposeCameraRow(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,)
    framenum: tuple[int, int]
    corners: np.ndarray
    ids: np.ndarray
    filled: np.ndarray




class CharucoObservation(BaseObservation):
    tracker_type: TrackerTypeString = 'charuco_tracker'
    all_charuco_ids: list[int]
    all_charuco_corners_in_object_coordinates: AllCharucoCorners3DByIdInObjectCoordinates

    all_aruco_ids: list[int]
    all_aruco_corners_in_object_coordinates: AllArucoCorners3DByIdInObjectCoordinates

    raw_charuco_corners: RawCharucoCornersImageCoordinates | None

    detected_charuco_corner_ids: DetectedCharucoCornerIds | None
    detected_charuco_corners_image_coordinates: DetectedCharucoCornersImageCoordinates | None
    detected_charuco_corners_in_object_coordinates: DetectedCharucoCornersInObjectCoordinates | None


    detected_aruco_marker_ids: DetectedArucoMarkerIds | None
    detected_aruco_marker_corners: DetectedArucoMarkerCorners | None

    charuco_board_translation_vector: CharucoBoardTranslationVector | None
    charuco_board_rotation_vector: CharucoBoardRotationVector | None

    image_size: tuple[int, int]

    @classmethod
    def from_detection_results(cls,
                                  frame_number: int,
                                  detected_charuco_corners: DetectedCharucoCornersImageCoordinates,
                                  detected_charuco_corner_ids: DetectedCharucoCornerIds,
                                  detected_aruco_marker_corners: Sequence[ArucoMarkerCorners],
                                  detected_aruco_marker_ids: DetectedArucoMarkerIds,
                                  all_charuco_ids: list[int],
                                  all_charuco_corners_in_object_coordinates: AllCharucoCorners3DByIdInObjectCoordinates,
                                  all_aruco_ids: list[int],
                                  all_aruco_corners_in_object_coordinates: AllArucoCorners3DByIdInObjectCoordinates,
                                  image_size: tuple[int, int]
                                  ):


        if detected_aruco_marker_ids is not None:
            # squeeze out singleton dimensions (i.e. a.shape = [2,1,3] -> np.squeeze(a).shape = [2,3])

            if detected_aruco_marker_ids.shape == (1, 1):
                # deal with special case where only one marker is detected
                detected_aruco_marker_ids = detected_aruco_marker_ids[0]
            else:
                detected_aruco_marker_ids = np.squeeze(detected_aruco_marker_ids)
            detected_aruco_marker_corners = tuple([np.squeeze(corner) for corner in detected_aruco_marker_corners])

        detected_charuco_corners_in_object_coordinates: DetectedCharucoCornersInObjectCoordinates | None = None
        reshaped_detected_charuco_corner_ids: DetectedCharucoCornerIds | None = None
        reshaped_detected_charuco_corners: DetectedCharucoCornersImageCoordinates | None = None
        if detected_charuco_corner_ids is not None:
            if detected_charuco_corner_ids.shape == (1, 1):
                reshaped_detected_charuco_corner_ids = detected_charuco_corner_ids[0]
                reshaped_detected_charuco_corners = detected_charuco_corners[0]
            else:
                reshaped_detected_charuco_corner_ids = np.squeeze(detected_charuco_corner_ids)
                reshaped_detected_charuco_corners = np.squeeze(detected_charuco_corners)

            detected_charuco_corners_in_object_coordinates = all_charuco_corners_in_object_coordinates[reshaped_detected_charuco_corner_ids, :]

        return cls(
            frame_number=frame_number,
            raw_charuco_corners=detected_charuco_corners,
            detected_charuco_corner_ids=reshaped_detected_charuco_corner_ids,
            detected_charuco_corners_image_coordinates=reshaped_detected_charuco_corners,
            detected_charuco_corners_in_object_coordinates=detected_charuco_corners_in_object_coordinates,
            detected_aruco_marker_ids=detected_aruco_marker_ids,
            detected_aruco_marker_corners=detected_aruco_marker_corners,
            all_charuco_ids=all_charuco_ids,
            all_aruco_ids=all_aruco_ids,
            all_charuco_corners_in_object_coordinates=all_charuco_corners_in_object_coordinates,
            all_aruco_corners_in_object_coordinates=all_aruco_corners_in_object_coordinates,
            charuco_board_translation_vector=None,
            charuco_board_rotation_vector=None,
            image_size=image_size,
        )

    @property
    def charuco_empty(self):
        return self.detected_charuco_corner_ids is None

    @property
    def aruco_empty(self):
        return self.detected_aruco_marker_ids is None

    @property
    def detected_charuco_corners_in_full_array(self) -> DetectedCharucoCorners2DInFullArray:
        """
        Returns the detected charuco corners in a full array, where the indices correspond to the charuco ids
        Non-detected corners are set to np.nan
        :return:
        """
        full_array = np.full((len(self.all_charuco_ids), 2), np.nan)
        if self.charuco_empty or self.detected_charuco_corner_ids is None or self.detected_charuco_corners_image_coordinates is None:
            return full_array
        for corner_index, corner_id in enumerate(self.detected_charuco_corner_ids):
            full_array[corner_id] = self.detected_charuco_corners_image_coordinates[corner_index]
        return full_array

    @property
    def charuco_corners_dict(self) -> dict[int, np.ndarray[2]]:
        corner_dict = {}
        if self.charuco_empty or self.detected_charuco_corner_ids is None or self.detected_charuco_corners_image_coordinates is None:
            return corner_dict
        for corner_index, corner_id in enumerate(self.detected_charuco_corner_ids):
            corner_dict[corner_id] = np.squeeze(self.detected_charuco_corners_image_coordinates[corner_index])
        return corner_dict

    @property
    def aruco_corners_dict(self) -> dict[int, np.ndarray[4, 2]]:
        corner_dict = {}
        if self.aruco_empty or self.detected_aruco_marker_ids is None or self.detected_aruco_marker_corners is None:
            return corner_dict
        for corner_index, corner_id in enumerate(self.detected_aruco_marker_ids):
            corner_dict[corner_id] = np.squeeze(self.detected_aruco_marker_corners[corner_index])
        return corner_dict
    
    def to_2d_array(self) -> DetectedCharucoCorners2DInFullArray:
        return self.detected_charuco_corners_in_full_array

    def to_tracked_points(self) -> dict[TrackedPointIdString, TrackedPoint2dArray]:

        """
        Converts the detected charuco corners to a dictionary of tracked points.
        The keys are the charuco ids and the values are the tracked points.
        :return: dict[TrackedPointIdString, TrackedPoint2d]
        """
        if self.charuco_empty or self.detected_charuco_corner_ids is None or self.detected_charuco_corners_image_coordinates is None:
            return {}
        tracked_points_2d: dict[TrackedPointIdString, TrackedPoint2dArray] = {}
        for charuco_corner_index in range(self.to_2d_array().shape[0]):
            point2d = self.to_2d_array()[charuco_corner_index]
            if np.isnan(point2d).any():
                continue

            tracked_points_2d[f"CharucoCorner-{charuco_corner_index}"] = point2d
        return tracked_points_2d

    def to_anipose_camera_row(self) -> dict[str, Any] | None:
        filled = np.full((len(self.all_charuco_ids), 1, 2), fill_value=np.nan)
        if self.charuco_empty or self.raw_charuco_corners is None or self.detected_charuco_corner_ids is None:
            return None
        for id, corner in zip(self.detected_charuco_corner_ids.ravel(), self.raw_charuco_corners):
            filled[id] = corner
        camera_row = AniposeCameraRow(
            framenum=(0, self.frame_number),
            corners=self.raw_charuco_corners,
            ids=self.detected_charuco_corner_ids,
            filled=filled,
        )
        return camera_row.model_dump()


CharucoObservations = list[CharucoObservation]
