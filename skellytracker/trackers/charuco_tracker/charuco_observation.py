from typing import Any, Sequence

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict
from numpydantic import NDArray, Shape

from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseObservation,
    TrackerTypeString,
    TrackedPoint2dArray,
    TrackedPointIdString
)
import logging
logger = logging.getLogger(__name__)

AllCharucoCorners3DByIdInObjectCoordinates = NDArray[Shape["* charuco_id, 3 xyz"], np.float32]
AllArucoCorners3DByIdInObjectCoordinates = NDArray[Shape["* aruco_ids, 4 corners, 3 xyz"], np.float32]
DetectedCharucoCornerIds = NDArray[Shape["* charuco_id, ..."], int]
RawCharucoCornersImageCoordinates = NDArray[Shape["* charuco_id,1 dim,  2 pxpy"], float]
DetectedCharucoCornersImageCoordinates = NDArray[Shape["* charuco_id, 2 pxpy"], float]
DetectedCharucoCornersInObjectCoordinates = NDArray[Shape["* charuco_id, 3 xyz"], float]
DetectedCharucoCornersInCameraCoordinates = NDArray[Shape["* charuco_id, 3 xyz"], float]

DetectedCharucoCorners2DInFullArray = NDArray[Shape["* charuco_id, 2 pxpy"], float]

ArucoMarkerCorners = NDArray[Shape["4 corners, 2 pxpy"], float]
DetectedArucoMarkerIds = NDArray[Shape["* aruco_id, ..."], int]
DetectedArucoMarkerCorners = Sequence[NDArray[Shape[" 4 corners, 2 pxpy"], float]]
DetectedArucoMarkersInCameraCoordinates = NDArray[Shape["* aruco_id, 4 corners, 3 xyz"], float]

CharucoBoardTranslationVector = NDArray[Shape["3 tx_ty_tz"], np.float32]
CharucoBoardRotationVector = NDArray[Shape["3 rx_ry_rz"], np.float32]

class AniposeCameraRow(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    framenum: tuple[int, int]
    corners: np.ndarray
    ids: np.ndarray
    filled: np.ndarray



MINIMUM_CHARUCO_CORNERS_FOR_VISIBILITY = 6
MINIMUM_CHARUCO_CORNERS_FOR_POSE = 6


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

    # Board pose in camera coordinates
    charuco_board_translation_vector: CharucoBoardTranslationVector | None = None
    charuco_board_rotation_vector: CharucoBoardRotationVector | None = None

    # Corners transformed to camera coordinates
    detected_charuco_corners_in_camera_coordinates: DetectedCharucoCornersInCameraCoordinates | None = None
    detected_aruco_markers_in_camera_coordinates: DetectedArucoMarkersInCameraCoordinates | None = None

    image_size: tuple[int, int]

    @property
    def charuco_board_visible(self) -> bool:
        if self.detected_charuco_corner_ids is None:
            return False
        return len(self.detected_charuco_corner_ids) >= MINIMUM_CHARUCO_CORNERS_FOR_VISIBILITY

    @property
    def has_board_pose(self) -> bool:
        """Check if board pose has been computed."""
        return (self.charuco_board_rotation_vector is not None and
                self.charuco_board_translation_vector is not None)

    def compute_board_pose_and_camera_coordinates(
            self,
            camera_matrix: np.ndarray,
            distortion_coefficients: np.ndarray,
    ) -> None:
        """
        Compute board pose in camera coordinates and transform all corners to camera space.

        This method uses cv2.solvePnP to estimate the board's position and orientation
        relative to the camera, then transforms all detected corners from board (object)
        coordinates to camera coordinates.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            distortion_coefficients: Camera distortion coefficients

        Raises:
            ValueError: If insufficient corners detected or pose estimation fails
        """
        if self.charuco_empty:
            logger.warning(
                f"Cannot compute board pose for frame {self.frame_number}: "
                "No charuco corners detected"
            )
            return

        if self.detected_charuco_corner_ids is None or self.detected_charuco_corners_image_coordinates is None:
            logger.warning(
                f"Cannot compute board pose for frame {self.frame_number}: "
                "Missing corner data"
            )
            return

        if len(self.detected_charuco_corner_ids) < MINIMUM_CHARUCO_CORNERS_FOR_POSE:
            logger.warning(
                f"Cannot compute board pose for frame {self.frame_number}: "
                f"Need at least {MINIMUM_CHARUCO_CORNERS_FOR_POSE} corners, "
                f"got {len(self.detected_charuco_corner_ids)}"
            )
            return

        if self.detected_charuco_corners_in_object_coordinates is None:
            logger.warning(
                f"Cannot compute board pose for frame {self.frame_number}: "
                "Missing object coordinates"
            )
            return

        # Estimate board pose using solvePnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            objectPoints=self.detected_charuco_corners_in_object_coordinates,
            imagePoints=self.detected_charuco_corners_image_coordinates,
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coefficients,
        )

        if not success:
            logger.warning(
                f"Failed to estimate board pose for frame {self.frame_number}"
            )
            return

        # Store board pose
        self.charuco_board_rotation_vector = np.squeeze(rotation_vector).astype(np.float32)
        self.charuco_board_translation_vector = np.squeeze(translation_vector).astype(np.float32)

        # Transform detected charuco corners to camera coordinates
        self.detected_charuco_corners_in_camera_coordinates = self._transform_points_to_camera_coordinates(
            points_object=self.detected_charuco_corners_in_object_coordinates,
            rotation_vector=self.charuco_board_rotation_vector,
            translation_vector=self.charuco_board_translation_vector,
        )

        # Transform detected aruco marker corners to camera coordinates
        if (not self.aruco_empty and
                self.detected_aruco_marker_ids is not None and
                self.detected_aruco_marker_corners is not None):

            # Get object coordinates for detected aruco markers
            aruco_object_coords_list = []
            for marker_id in self.detected_aruco_marker_ids:
                if marker_id not in self.all_aruco_ids:
                    continue
                marker_idx = self.all_aruco_ids.index(marker_id)
                aruco_object_coords_list.append(
                    self.all_aruco_corners_in_object_coordinates[marker_idx]
                )

            aruco_object_coords = np.array(aruco_object_coords_list, dtype=np.float32)

            # Reshape to (n_markers * 4, 3) for transformation
            n_markers = aruco_object_coords.shape[0]
            aruco_flat = aruco_object_coords.reshape(-1, 3)

            # Transform to camera coordinates
            aruco_camera_flat = self._transform_points_to_camera_coordinates(
                points_object=aruco_flat,
                rotation_vector=self.charuco_board_rotation_vector,
                translation_vector=self.charuco_board_translation_vector,
            )

            # Reshape back to (n_markers, 4, 3)
            self.detected_aruco_markers_in_camera_coordinates = aruco_camera_flat.reshape(
                n_markers, 4, 3
            )

    @staticmethod
    def _transform_points_to_camera_coordinates(
            points_object: np.ndarray,
            rotation_vector: np.ndarray,
            translation_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Transform points from object coordinates to camera coordinates.

        Uses Rodrigues formula to convert rotation vector to rotation matrix,
        then applies: point_camera = R @ point_object + t

        Args:
            points_object: Nx3 array of points in object coordinates
            rotation_vector: 3-element rotation vector
            translation_vector: 3-element translation vector

        Returns:
            Nx3 array of points in camera coordinates
        """
        if points_object.shape[1] != 3:
            raise ValueError(
                f"Expected points with shape (N, 3), got {points_object.shape}"
            )

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Transform: p_camera = R @ p_object + t
        # Broadcasting handles the translation vector addition
        points_camera = (rotation_matrix @ points_object.T).T + translation_vector

        return points_camera.astype(np.float32)

    @classmethod
    def from_detection_results(
            cls,
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

            detected_charuco_corners_in_object_coordinates = all_charuco_corners_in_object_coordinates[
                reshaped_detected_charuco_corner_ids, :
            ]

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
            detected_charuco_corners_in_camera_coordinates=None,
            detected_aruco_markers_in_camera_coordinates=None,
            image_size=image_size,
        )

    @property
    def charuco_empty(self) -> bool:
        return self.detected_charuco_corner_ids is None

    @property
    def aruco_empty(self) -> bool:
        return self.detected_aruco_marker_ids is None

    @property
    def detected_charuco_corners_in_full_array(self) -> DetectedCharucoCorners2DInFullArray:
        """
        Returns the detected charuco corners in a full array, where the indices correspond to the charuco ids
        Non-detected corners are set to np.nan
        """
        full_array = np.full((len(self.all_charuco_ids), 2), np.nan)
        if (self.charuco_empty or
                self.detected_charuco_corner_ids is None or
                self.detected_charuco_corners_image_coordinates is None):
            return full_array
        for corner_index, corner_id in enumerate(self.detected_charuco_corner_ids):
            full_array[corner_id] = self.detected_charuco_corners_image_coordinates[corner_index]
        return full_array

    @property
    def charuco_corners_dict(self) -> dict[int, np.ndarray]:
        corner_dict = {}
        if (self.charuco_empty or
                self.detected_charuco_corner_ids is None or
                self.detected_charuco_corners_image_coordinates is None):
            return corner_dict
        for corner_index, corner_id in enumerate(self.detected_charuco_corner_ids):
            corner_dict[corner_id] = np.squeeze(self.detected_charuco_corners_image_coordinates[corner_index])
        return corner_dict

    @property
    def aruco_corners_dict(self) -> dict[int, np.ndarray]:
        corner_dict = {}
        if (self.aruco_empty or
                self.detected_aruco_marker_ids is None or
                self.detected_aruco_marker_corners is None):
            return corner_dict
        for corner_index, corner_id in enumerate(self.detected_aruco_marker_ids):
            corner_dict[corner_id] = np.squeeze(self.detected_aruco_marker_corners[corner_index])
        return corner_dict

    def to_2d_array(self, *, confidence_threshold: float | None = None,
                    fill_with_nans: bool = True) -> DetectedCharucoCorners2DInFullArray:
        """
        Convert to 2D array. Confidence filtering not supported for Charuco.

        Args:
            confidence_threshold: Ignored for Charuco tracker.
            fill_with_nans: Ignored for Charuco tracker.
        """
        if confidence_threshold is not None:
            logger.warning(
                "Confidence filtering requested but not supported for Charuco tracker. Returning all detected points.")

        return self.detected_charuco_corners_in_full_array

    def to_tracked_points(self, *, confidence_threshold: float | None = None) -> dict[
        TrackedPointIdString, TrackedPoint2dArray]:
        """
        Get tracked points. Confidence filtering not supported for Charuco.

        Args:
            confidence_threshold: Ignored for Charuco tracker.
        """
        if confidence_threshold is not None:
            logger.warning(
                "Confidence filtering requested but not supported for Charuco tracker. Returning all detected points.")

        if (self.charuco_empty or
                self.detected_charuco_corner_ids is None or
                self.detected_charuco_corners_image_coordinates is None):
            return {}

        tracked_points_2d: dict[TrackedPointIdString, TrackedPoint2dArray] = {}
        for charuco_corner_index in range(self.to_2d_array().shape[0]):
            point2d = self.to_2d_array()[charuco_corner_index]
            if np.isnan(point2d).any():
                continue
            tracked_points_2d[f"CharucoCorner-{charuco_corner_index}"] = point2d

        return tracked_points_2d

    def to_anipose_camera_row(self) -> dict[str, Any] | None:
        nan_filled = np.full((len(self.all_charuco_ids), 1, 2), fill_value=np.nan)
        if self.charuco_empty or self.raw_charuco_corners is None or self.detected_charuco_corner_ids is None:
            nan_row = AniposeCameraRow(
                framenum=(0, self.frame_number),
                corners=nan_filled,
                ids=np.asarray(self.all_charuco_ids),
                filled=nan_filled,
            )
            return nan_row.model_dump()
        for id, corner in zip(self.detected_charuco_corner_ids.ravel(), self.raw_charuco_corners):
            nan_filled[id] = corner
        camera_row = AniposeCameraRow(
            framenum=(0, self.frame_number),
            corners=self.raw_charuco_corners,
            ids=self.detected_charuco_corner_ids,
            filled=nan_filled,
        )
        return camera_row.model_dump()


CharucoObservations = list[CharucoObservation]