import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation
from skellytracker.trackers.base_tracker.point_cloud import PointCloud

logger = logging.getLogger(__name__)

# Type aliases for documentation clarity
AllCharucoCorners3DByIdInObjectCoordinates = NDArray[np.float32]
AllArucoCorners3DByIdInObjectCoordinates = NDArray[np.float32]
DetectedCharucoCornerIds = NDArray[np.integer]
RawCharucoCornersImageCoordinates = NDArray[np.floating]
DetectedCharucoCornersImageCoordinates = NDArray[np.floating]
DetectedCharucoCornersInObjectCoordinates = NDArray[np.floating]
DetectedCharucoCornersInCameraCoordinates = NDArray[np.floating]
ArucoMarkerCorners = NDArray[np.floating]
DetectedArucoMarkerIds = NDArray[np.integer]
DetectedArucoMarkerCorners = Sequence[NDArray[np.floating]]
DetectedArucoMarkersInCameraCoordinates = NDArray[np.floating]
CharucoBoardTranslationVector = NDArray[np.float32]
CharucoBoardRotationVector = NDArray[np.float32]


@dataclass
class AniposeCameraRow:
    framenum: tuple[int, int]
    corners: np.ndarray
    ids: np.ndarray
    filled: np.ndarray

    def to_dict(self) -> dict:
        return {
            "framenum": self.framenum,
            "corners": self.corners,
            "ids": self.ids,
            "filled": self.filled,
        }


MINIMUM_CHARUCO_CORNERS_FOR_VISIBILITY = 6
MINIMUM_CHARUCO_CORNERS_FOR_POSE = 6


@dataclass
class CharucoObservation(BaseObservation):
    """
    Charuco board observation.

    The PointCloud contains all charuco corner positions in a fixed-size
    array indexed by corner ID. Undetected corners are NaN.

    NOTE: Not using slots=True because compute_board_pose_and_camera_coordinates
    mutates optional fields that may not be set at construction.
    """

    tracker_type: str = field(default="charuco_tracker", init=False)
    frame_number: int = 0
    image_size: tuple[int, int] = (0, 0)

    # The PointCloud: one row per charuco corner ID, NaN if not detected
    points: PointCloud = field(default_factory=lambda: PointCloud.empty(("empty",)))

    # Board definition
    all_charuco_ids: list[int] = field(default_factory=list)
    all_charuco_corners_in_object_coordinates: AllCharucoCorners3DByIdInObjectCoordinates | None = None
    all_aruco_ids: list[int] = field(default_factory=list)
    all_aruco_corners_in_object_coordinates: AllArucoCorners3DByIdInObjectCoordinates | None = None

    # Raw detection data
    raw_charuco_corners: RawCharucoCornersImageCoordinates | None = None
    detected_charuco_corner_ids: DetectedCharucoCornerIds | None = None
    detected_charuco_corners_image_coordinates: DetectedCharucoCornersImageCoordinates | None = None
    detected_charuco_corners_in_object_coordinates: DetectedCharucoCornersInObjectCoordinates | None = None
    detected_aruco_marker_ids: DetectedArucoMarkerIds | None = None
    detected_aruco_marker_corners: DetectedArucoMarkerCorners | None = None

    # Board pose (computed after construction by compute_board_pose_and_camera_coordinates)
    charuco_board_translation_vector: CharucoBoardTranslationVector | None = None
    charuco_board_rotation_vector: CharucoBoardRotationVector | None = None
    detected_charuco_corners_in_camera_coordinates: DetectedCharucoCornersInCameraCoordinates | None = None
    detected_aruco_markers_in_camera_coordinates: DetectedArucoMarkersInCameraCoordinates | None = None

    @classmethod
    def from_detection_results(
            cls,
            frame_number: int,
            detected_charuco_corners: DetectedCharucoCornersImageCoordinates | None,
            detected_charuco_corner_ids: DetectedCharucoCornerIds | None,
            detected_aruco_marker_corners: Sequence[ArucoMarkerCorners] | None,
            detected_aruco_marker_ids: DetectedArucoMarkerIds | None,
            all_charuco_ids: list[int],
            all_charuco_corners_in_object_coordinates: AllCharucoCorners3DByIdInObjectCoordinates,
            all_aruco_ids: list[int],
            all_aruco_corners_in_object_coordinates: AllArucoCorners3DByIdInObjectCoordinates,
            image_size: tuple[int, int],
    ) -> "CharucoObservation":
        if detected_aruco_marker_ids is not None:
            if detected_aruco_marker_ids.shape == (1, 1):
                detected_aruco_marker_ids = detected_aruco_marker_ids[0]
            else:
                detected_aruco_marker_ids = np.squeeze(detected_aruco_marker_ids)
            detected_aruco_marker_corners = tuple([np.squeeze(corner) for corner in detected_aruco_marker_corners])

        detected_charuco_corners_in_object_coordinates: DetectedCharucoCornersInObjectCoordinates | None = None
        reshaped_detected_charuco_corner_ids: DetectedCharucoCornerIds | None = None
        reshaped_detected_charuco_corners: DetectedCharucoCornersImageCoordinates | None = None
        if detected_charuco_corner_ids is not None:
            if detected_charuco_corners is None:
                raise ValueError(
                    f"Frame {frame_number}: detected_charuco_corner_ids is non-None but detected_charuco_corners is None"
                )
            if detected_charuco_corner_ids.shape == (1, 1):
                reshaped_detected_charuco_corner_ids = detected_charuco_corner_ids[0]
                reshaped_detected_charuco_corners = detected_charuco_corners[0]
            else:
                reshaped_detected_charuco_corner_ids = np.squeeze(detected_charuco_corner_ids)
                reshaped_detected_charuco_corners = np.squeeze(detected_charuco_corners)

            detected_charuco_corners_in_object_coordinates = all_charuco_corners_in_object_coordinates[
                reshaped_detected_charuco_corner_ids, :
            ]

        # Build the PointCloud from detected corners in full-array format.
        # Layout: [CharucoCorner-0..N-1, ArucoCorner-{id}-0..3 per marker in all_aruco_ids order]
        n_charuco = len(all_charuco_ids)
        n_aruco = len(all_aruco_ids)
        n_total = n_charuco + n_aruco * 4

        charuco_names = tuple(f"CharucoCorner-{i}" for i in range(n_charuco))
        aruco_names = tuple(
            f"ArucoCorner-{marker_id}-{c}"
            for marker_id in all_aruco_ids
            for c in range(4)
        )
        corner_names = charuco_names + aruco_names

        full_array_2d = np.full((n_total, 2), np.nan)
        visibility = np.zeros(n_total)

        if reshaped_detected_charuco_corner_ids is not None and reshaped_detected_charuco_corners is not None:
            for corner_index, corner_id in enumerate(reshaped_detected_charuco_corner_ids):
                full_array_2d[corner_id] = reshaped_detected_charuco_corners[corner_index]
                visibility[corner_id] = 1.0

        if (detected_aruco_marker_ids is not None and
                detected_aruco_marker_corners is not None and
                n_aruco > 0):
            aruco_id_to_slot = {mid: slot for slot, mid in enumerate(all_aruco_ids)}
            for marker_index, marker_id in enumerate(np.atleast_1d(detected_aruco_marker_ids)):
                slot = aruco_id_to_slot.get(int(marker_id))
                if slot is None:
                    continue
                marker_corners = np.asarray(detected_aruco_marker_corners[marker_index])
                if marker_corners.shape != (4, 2):
                    continue
                base = n_charuco + slot * 4
                full_array_2d[base:base + 4] = marker_corners
                visibility[base:base + 4] = 1.0

        # PointCloud stores (N, 3) — use z=0 for 2D image-space corners
        xyz = np.column_stack([full_array_2d, np.zeros(n_total)])
        cloud = PointCloud(names=corner_names, xyz=xyz, visibility=visibility)

        return cls(
            frame_number=frame_number,
            image_size=image_size,
            points=cloud,
            all_charuco_ids=all_charuco_ids,
            all_charuco_corners_in_object_coordinates=all_charuco_corners_in_object_coordinates,
            all_aruco_ids=all_aruco_ids,
            all_aruco_corners_in_object_coordinates=all_aruco_corners_in_object_coordinates,
            raw_charuco_corners=detected_charuco_corners,
            detected_charuco_corner_ids=reshaped_detected_charuco_corner_ids,
            detected_charuco_corners_image_coordinates=reshaped_detected_charuco_corners,
            detected_charuco_corners_in_object_coordinates=detected_charuco_corners_in_object_coordinates,
            detected_aruco_marker_ids=detected_aruco_marker_ids,
            detected_aruco_marker_corners=detected_aruco_marker_corners,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def charuco_empty(self) -> bool:
        return self.detected_charuco_corner_ids is None

    @property
    def aruco_empty(self) -> bool:
        return self.detected_aruco_marker_ids is None

    @property
    def charuco_board_visible(self) -> bool:
        if self.detected_charuco_corner_ids is None:
            return False
        return len(self.detected_charuco_corner_ids) >= MINIMUM_CHARUCO_CORNERS_FOR_VISIBILITY

    @property
    def has_board_pose(self) -> bool:
        return (self.charuco_board_rotation_vector is not None and
                self.charuco_board_translation_vector is not None)

    @property
    def detected_charuco_corners_in_full_array(self) -> NDArray:
        """Full array of charuco corners indexed by ID, NaN for undetected."""
        n_charuco = len(self.all_charuco_ids)
        return self.points.xy[:n_charuco].copy()

    @property
    def charuco_corners_dict(self) -> dict[int, np.ndarray]:
        corner_dict: dict[int, np.ndarray] = {}
        if self.charuco_empty or self.detected_charuco_corner_ids is None or self.detected_charuco_corners_image_coordinates is None:
            return corner_dict
        for corner_index, corner_id in enumerate(self.detected_charuco_corner_ids):
            corner_dict[int(corner_id)] = np.squeeze(self.detected_charuco_corners_image_coordinates[corner_index])
        return corner_dict

    @property
    def aruco_corners_dict(self) -> dict[int, np.ndarray]:
        corner_dict: dict[int, np.ndarray] = {}
        if self.aruco_empty or self.detected_aruco_marker_ids is None or self.detected_aruco_marker_corners is None:
            return corner_dict
        for corner_index, corner_id in enumerate(self.detected_aruco_marker_ids):
            corner_dict[int(corner_id)] = np.squeeze(self.detected_aruco_marker_corners[corner_index])
        return corner_dict

    # =========================================================================
    # Board pose computation (mutates self)
    # =========================================================================

    def compute_board_pose_and_camera_coordinates(
            self,
            camera_matrix: np.ndarray,
            distortion_coefficients: np.ndarray,
    ) -> None:
        """Compute board pose and transform corners to camera coordinates."""
        if self.charuco_empty:
            logger.warning(f"Cannot compute board pose for frame {self.frame_number}: No charuco corners detected")
            return

        if self.detected_charuco_corner_ids is None or self.detected_charuco_corners_image_coordinates is None:
            logger.warning(f"Cannot compute board pose for frame {self.frame_number}: Missing corner data")
            return

        if len(self.detected_charuco_corner_ids) < MINIMUM_CHARUCO_CORNERS_FOR_POSE:
            logger.warning(
                f"Cannot compute board pose for frame {self.frame_number}: "
                f"Need at least {MINIMUM_CHARUCO_CORNERS_FOR_POSE} corners, got {len(self.detected_charuco_corner_ids)}"
            )
            return

        if self.detected_charuco_corners_in_object_coordinates is None:
            logger.warning(f"Cannot compute board pose for frame {self.frame_number}: Missing object coordinates")
            return

        success, rotation_vector, translation_vector = cv2.solvePnP(
            objectPoints=self.detected_charuco_corners_in_object_coordinates,
            imagePoints=self.detected_charuco_corners_image_coordinates,
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coefficients,
        )

        if not success:
            logger.warning(f"Failed to estimate board pose for frame {self.frame_number}")
            return

        self.charuco_board_rotation_vector = np.squeeze(rotation_vector).astype(np.float32)
        self.charuco_board_translation_vector = np.squeeze(translation_vector).astype(np.float32)

        self.detected_charuco_corners_in_camera_coordinates = self._transform_points_to_camera_coordinates(
            points_object=self.detected_charuco_corners_in_object_coordinates,
            rotation_vector=self.charuco_board_rotation_vector,
            translation_vector=self.charuco_board_translation_vector,
        )

        if (not self.aruco_empty and
                self.detected_aruco_marker_ids is not None and
                self.detected_aruco_marker_corners is not None):
            aruco_object_coords_list = []
            for marker_id in self.detected_aruco_marker_ids:
                if marker_id not in self.all_aruco_ids:
                    continue
                marker_idx = self.all_aruco_ids.index(marker_id)
                aruco_object_coords_list.append(
                    self.all_aruco_corners_in_object_coordinates[marker_idx]
                )

            if aruco_object_coords_list:
                aruco_object_coords = np.array(aruco_object_coords_list, dtype=np.float32)
                n_markers = aruco_object_coords.shape[0]
                aruco_flat = aruco_object_coords.reshape(-1, 3)
                aruco_camera_flat = self._transform_points_to_camera_coordinates(
                    points_object=aruco_flat,
                    rotation_vector=self.charuco_board_rotation_vector,
                    translation_vector=self.charuco_board_translation_vector,
                )
                self.detected_aruco_markers_in_camera_coordinates = aruco_camera_flat.reshape(n_markers, 4, 3)

    @staticmethod
    def _transform_points_to_camera_coordinates(
            points_object: np.ndarray,
            rotation_vector: np.ndarray,
            translation_vector: np.ndarray,
    ) -> np.ndarray:
        """Transform points from object coordinates to camera coordinates."""
        if points_object.shape[1] != 3:
            raise ValueError(f"Expected points with shape (N, 3), got {points_object.shape}")
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        points_camera = (rotation_matrix @ points_object.T).T + translation_vector
        return points_camera.astype(np.float32)

    # =========================================================================
    # Anipose export
    # =========================================================================

    def to_anipose_camera_row(self) -> dict[str, Any] | None:
        nan_filled = np.full((len(self.all_charuco_ids), 1, 2), fill_value=np.nan)
        if self.charuco_empty or self.raw_charuco_corners is None or self.detected_charuco_corner_ids is None:
            nan_row = AniposeCameraRow(
                framenum=(0, self.frame_number),
                corners=nan_filled,
                ids=np.asarray(self.all_charuco_ids),
                filled=nan_filled,
            )
            return nan_row.to_dict()
        for id, corner in zip(self.detected_charuco_corner_ids.ravel(), self.raw_charuco_corners):
            nan_filled[id] = corner
        camera_row = AniposeCameraRow(
            framenum=(0, self.frame_number),
            corners=self.raw_charuco_corners,
            ids=self.detected_charuco_corner_ids,
            filled=nan_filled,
        )
        return camera_row.to_dict()


CharucoObservations = list[CharucoObservation]
