from dataclasses import dataclass, field

import cv2
import numpy as np

from skellytracker.trackers.charuco_tracker.charuco_observation import CharucoObservation, CharucoObservations


@dataclass
class CameraCalibrationError:
    mean_reprojection_error: float
    reprojection_error_by_view: list[float]
    jacobian: list[np.ndarray]


DEFAULT_INTRINSICS_COEFFICIENTS_COUNT = 5


@dataclass
class CameraCalibrationEstimator:
    """
    CameraCalibrationEstimator class for estimating camera calibration parameters.

    Attributes:
        camera_matrix (np.ndarray[3, 3]): 3x3 matrix with rows: [focal_length_x, 0, center_point_x],
                                           [0, focal_length_y, center_point_y], [0, 0, 1]
        distortion_coefficients (np.ndarray[..., 1]): Distortion coefficients to include, e.g. [k1, k2, p1, p2, k3].
                                                      Can be 4, 5, 8, 12, or 14 elements.
        object_points_views (list[np.ndarray[..., 3]]): List of 3D points in the world (or object) coordinate space.
        image_points_views (list[np.ndarray[..., 2]]): List of 2D points in the image coordinate space (i.e. pixel coordinates).
        rotation_vectors (list[np.ndarray[..., 3]]): List of rotation vectors, containing the rotation of the camera
                                                     relative to the world coordinate space for each view.
        translation_vectors (list[np.ndarray[..., 3]]): List of translation vectors, containing the translation of the camera
                                                        relative to the world coordinate space for each view.
        error (CameraCalibrationError | None): Calibration error details.
    """
    image_size: tuple[int, ...]

    charuco_corner_ids: list[int]
    charuco_corners_in_object_coordinates: np.ndarray[..., 3]

    aruco_marker_ids: list[int]
    aruco_corners_in_object_coordinates: list[np.ndarray[..., 3]]

    distortion_coefficients: np.ndarray[..., 1]
    camera_matrix: np.ndarray[3, 3]

    object_points_views: list[np.ndarray[..., 3]] = field(default_factory=list)
    image_points_views: list[np.ndarray[..., 2]] = field(default_factory=list)
    rotation_vectors: list[np.ndarray[..., 3]] = field(default_factory=list)
    translation_vectors: list[np.ndarray[..., 3]] = field(default_factory=list)
    error: CameraCalibrationError | None = None

    @classmethod
    def create_initial(cls,
                       image_size: tuple[int, ...],
                       aruco_marker_ids: list[int],
                       aruco_corners_in_object_coordinates: list[np.ndarray[..., 3]],
                       charuco_corner_ids: list[int],
                       charuco_corners_in_object_coordinates: np.ndarray[..., 3],
                       number_of_distortion_coefficients: int = DEFAULT_INTRINSICS_COEFFICIENTS_COUNT):
        camera_matrix = np.eye(3)
        camera_matrix[0, 2] = image_size[0] / 2  # x_center
        camera_matrix[1, 2] = image_size[1] / 2  # y_center

        if not number_of_distortion_coefficients in [4, 5, 8, 12, 14]:
            raise ValueError("Invalid number of distortion coefficients. Must be 4, 5, 8, 12, or 14.")

        if len(charuco_corner_ids) != charuco_corners_in_object_coordinates.shape[0]:
            raise ValueError("Number of charuco corner IDs must match the number of charuco corners.")
        if len(aruco_marker_ids) != len(aruco_corners_in_object_coordinates):
            raise ValueError("Number of aruco marker IDs must match the number of aruco corners.")

        return cls(image_size=image_size,
                   charuco_corner_ids=charuco_corner_ids,
                   charuco_corners_in_object_coordinates=charuco_corners_in_object_coordinates,
                   aruco_marker_ids=aruco_marker_ids,
                   aruco_corners_in_object_coordinates=aruco_corners_in_object_coordinates,
                   camera_matrix=camera_matrix,
                   distortion_coefficients=np.zeros(number_of_distortion_coefficients),
                   )

    charuco_observations: CharucoObservations = field(default_factory=list)

    def add_observation(self, observation: CharucoObservation):
        if observation.charuco_empty:
            return
        self._validate_observation(observation)

        self.charuco_observations.append(observation)
        self.image_points_views.append(observation.detected_charuco_corners_image_coordinates)
        self.object_points_views.append(self.charuco_corners_in_object_coordinates)

    def _validate_observation(self, observation:CharucoObservation):
        if observation.image_size != self.image_size:
            raise ValueError("Image size mismatch")
        if any([corner_id not in self.charuco_corner_ids for corner_id in observation.detected_charuco_corner_ids]):
            raise ValueError(
                f"Invalid charuco corner ID detected: {observation.detected_charuco_corner_ids} not all in {self.charuco_corner_ids}")


    def update_calibration_estimate(self):
        if len(self.object_points_views) < len(self.charuco_corner_ids):
            raise ValueError(f"You must have at least as many observations as charuco corners: "
                             f"#Current views: {len(self.object_points_views)}, "
                             f"#Charuco corners: {len(self.charuco_corner_ids)}")

        # https://docs.opencv.org/4.10.0/d9/d0c/group__calib3d.html#ga687a1ab946686f0d85ae0363b5af1d7b
        (success,
         camera_matrix,
         distortion_coefficients,
         rotation_vectors,
         translation_vectors) = cv2.calibrateCamera(objectPoints=self.object_points_views,
                                                    imagePoints=self.image_points_views,
                                                    imageSize=self.image_size,
                                                    cameraMatrix=self.camera_matrix,
                                                    distCoeffs=self.distortion_coefficients,
                                                    )

        if not success:
            raise ValueError(f"Camera Calibration failed! Check your input data:",
                                f"object_points_views: {self.object_points_views}",
                                f"image_points_views: {self.image_points_views}",
                                f"camera_matrix: {self.camera_matrix}",
                                f"distortion_coefficients: {self.distortion_coefficients}",
                                )
    #     self._update_reprojection_error()
    #
    # def _update_reprojection_error(self) -> CameraCalibrationError:
    #     if len(self.image_points_views) != len(self.object_points_views):
    #         raise ValueError("The number of image and object points must be the same")
    #     if len(self.image_points_views) == 0:
    #         raise ValueError("No image points provided")
    #     error = None
    #     jacobian = None
    #     reprojection_error_by_view = []
    #     for view_index in range(len(estimate.image_points_views)):
    #         projected_image_points, jacobian = cv2.projectPoints(estimate.object_points_views[view_index],
    #                                                              estimate.rotation_vectors[view_index],
    #                                                              estimate.translation_vectors[view_index],
    #                                                              estimate.camera_matrix,
    #                                                              estimate.distortion_coefficients)
    #
    #         reprojection_error_by_view.append(cv2.norm(estimate.image_points_views[view_index],
    #                                                    projected_image_points, cv2.NORM_L2) / len(
    #             projected_image_points))
    #
    #     mean_error = np.mean(reprojection_error_by_view) if reprojection_error_by_view else -1
    #     return CameraCalibrationError(mean_reprojection_error=mean_error,
    #                                   reprojection_error_by_view=reprojection_error_by_view,
    #                                   jacobian=[jacobian])
