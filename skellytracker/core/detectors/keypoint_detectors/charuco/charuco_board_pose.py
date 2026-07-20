from __future__ import annotations

import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from skellytracker.core.data_primitives import Keypoints
from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_board_definition import (
    CharucoBoardDefinition,
)

logger = logging.getLogger(__name__)

MINIMUM_CHARUCO_CORNERS_FOR_POSE = 6


def compute_board_pose(
    keypoints: Keypoints,
    board_def: CharucoBoardDefinition,
    camera_matrix: NDArray[np.float64],
    distortion_coefficients: NDArray[np.float64],
) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
    """Estimate board pose from detected charuco corners via solvePnP.

    Returns (rotation_vector, translation_vector) or None if pose cannot be estimated.
    Requires at least MINIMUM_CHARUCO_CORNERS_FOR_POSE detected corners.
    """
    object_points, image_points = _collect_detected_charuco_points(keypoints, board_def)

    if object_points is None or len(object_points) < MINIMUM_CHARUCO_CORNERS_FOR_POSE:
        n_detected = 0 if object_points is None else len(object_points)
        logger.warning(
            f"Cannot estimate board pose: need {MINIMUM_CHARUCO_CORNERS_FOR_POSE} corners, "
            f"got {n_detected}"
        )
        return None

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion_coefficients,
    )
    if not success:
        logger.warning("solvePnP failed to estimate board pose")
        return None

    return np.squeeze(rvec).astype(np.float32), np.squeeze(tvec).astype(np.float32)


def transform_to_camera_coordinates(
    points_object: NDArray[np.floating],
    rvec: NDArray[np.float32],
    tvec: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Transform object-frame points (N, 3) to camera-frame coordinates."""
    if points_object.ndim != 2 or points_object.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {points_object.shape}")
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    return ((rotation_matrix @ points_object.T).T + tvec).astype(np.float32)


def _collect_detected_charuco_points(
    keypoints: Keypoints,
    board_def: CharucoBoardDefinition,
) -> tuple[NDArray[np.float32], NDArray[np.float32]] | tuple[None, None]:
    """Extract detected corner image coords and matching object-frame coords."""
    object_points_list = []
    image_points_list = []
    board_frame = board_def.corner_positions_board_frame  # (n_corners, 3)

    for corner_id in range(board_def.n_corners):
        name = f"CharucoCorner-{corner_id}"
        if not keypoints.has_name(name):
            continue
        idx = keypoints.index_of(name)
        if keypoints.visibility[idx] == 0.0 or np.isnan(keypoints.xyz[idx, 0]):
            continue
        object_points_list.append(board_frame[corner_id])
        image_points_list.append(keypoints.xyz[idx, :2])

    if not object_points_list:
        return None, None

    return (
        np.array(object_points_list, dtype=np.float32),
        np.array(image_points_list, dtype=np.float32),
    )
