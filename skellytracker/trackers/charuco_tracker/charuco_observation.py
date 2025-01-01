from dataclasses import dataclass

import numpy as np

from skellytracker.trackers.base_tracker.base_tracker import BaseObservation


@dataclass
class CharucoObservation(BaseObservation):
    all_charuco_ids: list[int]
    all_charuco_corners_in_object_coordinates: np.ndarray[..., 3]
    detected_charuco_corner_ids: list[list[int]]
    detected_charuco_corners_image_coordinates: np.ndarray[..., 2]
    detected_charuco_corners_in_object_coordinates: np.ndarray[..., 3]

    all_aruco_ids: list[int]
    all_aruco_corners_in_object_coordinates: np.ndarray[..., 3]
    detected_aruco_marker_ids: list[list[int]]
    detected_aruco_marker_corners: tuple[np.ndarray[..., 2]]



    image_size: tuple[int, int]

    @classmethod
    def from_detect_board_results(cls,
                                  detected_charuco_corners: np.ndarray,
                                  detected_charuco_corner_ids: list[list[int]],
                                  detected_aruco_marker_corners: tuple[np.ndarray[..., 2]],
                                  detected_aruco_marker_ids: list[list[int]],
                                  all_charuco_ids: list[int],
                                  all_charuco_corners_in_object_coordinates: np.ndarray[..., 3],
                                  all_aruco_ids: list[int],
                                  all_aruco_corners_in_object_coordinates: np.ndarray[..., 3],
                                  image_size: tuple[int, int]):

        detected_charuco_corners_in_object_coordinates = all_charuco_corners_in_object_coordinates[np.squeeze(detected_charuco_corner_ids), :] if detected_charuco_corner_ids is not None else None
        return cls(
            detected_charuco_corner_ids=detected_charuco_corner_ids,
            detected_charuco_corners_image_coordinates=detected_charuco_corners,
            detected_charuco_corners_in_object_coordinates=detected_charuco_corners_in_object_coordinates,
            detected_aruco_marker_ids=detected_aruco_marker_ids,
            detected_aruco_marker_corners=detected_aruco_marker_corners,
            all_charuco_ids=all_charuco_ids,
            all_aruco_ids=all_aruco_ids,
            all_charuco_corners_in_object_coordinates=all_charuco_corners_in_object_coordinates,
            all_aruco_corners_in_object_coordinates=all_aruco_corners_in_object_coordinates,
            image_size=image_size
        )

    @property
    def charuco_empty(self):
        return self.detected_charuco_corner_ids is None

    @property
    def aruco_empty(self):
        return self.detected_aruco_marker_ids is None

    @property
    def charuco_corners_dict(self) -> dict[int, np.ndarray[2]]:
        corner_dict = {}
        if self.charuco_empty:
            return corner_dict
        for corner_index, corner_id in enumerate(self.detected_charuco_corner_ids):
            corner_dict[corner_id[0]] = np.squeeze(self.detected_charuco_corners_image_coordinates[corner_index])
        return corner_dict

    @property
    def aruco_corners_dict(self) -> dict[int, np.ndarray[4, 2]]:
        corner_dict = {}
        if self.aruco_empty:
            return corner_dict
        for corner_index, corner_id in enumerate(self.detected_aruco_marker_ids):
            corner_dict[corner_id[0]] = np.squeeze(self.detected_aruco_marker_corners[corner_index])
        return corner_dict


CharucoObservations = list[CharucoObservation]
