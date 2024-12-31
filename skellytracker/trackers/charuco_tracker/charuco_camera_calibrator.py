from dataclasses import dataclass
from typing import Sequence, Any

import cv2
import numpy as np

from skellytracker.trackers.charuco_tracker.charuco_observations import CharucoObservation


@dataclass
class CameraCalibrationError:
    total_reprojection_error: float
    reprojection_error_by_point: list[float]
    jacobian: list[np.ndarray]


@dataclass
class CameraCalibrationEstimate:
    object_points: np.ndarray
    image_points: np.ndarray
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rotation_vectors: Sequence[np.ndarray | Any]
    translation_vectors: Sequence[np.ndarray | Any]
    reprojection_error: CameraCalibrationError

    @classmethod
    def from_observation(cls, observation:CharucoObservation):
        charuco_corners_image_coordinates = observation.charuco_corners

        success, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(
            objectPoints= charuco_corners_object_coordinates,
            imagePoints=charuco_corners_image_coordinates,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None
        )

        return CameraCalibrationEstimate(object_points=charuco_corners_object_coordinates,
                                         image_points=charuco_corners_image_coordinates,
                                         camera_matrix=camera_matrix,
                                         distortion_coefficients=distortion_coefficients,
                                         rotation_vectors=rotation_vectors,
                                         translation_vectors=translation_vectors,
                                         reprojection_error=self._calculate_reprojection_error(
                                             charuco_corners_image_coordinates=[charuco_corners_image_coordinates],
                                             charuco_corners_object_coordinates=[charuco_corners_object_coordinates],
                                             camera_matrix=camera_matrix,
                                             distortion_coefficients=distortion_coefficients,
                                             rotation_vectors=rotation_vectors,
                                             translation_vectors=translation_vectors)
                                         )

    def _calculate_reprojection_error(self,
                                      charuco_corners_image_coordinates: list[np.ndarray],
                                      charuco_corners_object_coordinates: list[np.ndarray],
                                      camera_matrix: np.ndarray,
                                      distortion_coefficients: np.ndarray,
                                      rotation_vectors: Sequence[np.ndarray | Any],
                                      translation_vectors: Sequence[np.ndarray | Any]
                                      ) -> CameraCalibrationError:

        if len(charuco_corners_image_coordinates) != len(charuco_corners_object_coordinates):
            raise ValueError("The number of image and object points must be the same")
        if len(charuco_corners_image_coordinates) == 0:
            raise ValueError("No image points provided")
        total_error = 0
        error = None
        jacobian = None
        for corner_index in range(len(charuco_corners_object_coordinates)):
            projected_image_points, jacobian = cv2.projectPoints(charuco_corners_object_coordinates[corner_index],
                                                                 rotation_vectors[corner_index],
                                                                 translation_vectors[corner_index],
                                                                 camera_matrix,
                                                                 distortion_coefficients)

            error = cv2.norm(charuco_corners_image_coordinates[corner_index], projected_image_points,
                             cv2.NORM_L2) / len(projected_image_points)
            total_error += error
        return CameraCalibrationError(total_reprojection_error=total_error,
                                      reprojection_error_by_point=[error],
                                      jacobian=[jacobian])
