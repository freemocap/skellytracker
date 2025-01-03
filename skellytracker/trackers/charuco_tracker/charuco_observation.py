import json
from dataclasses import dataclass

import numpy as np

from skellytracker.trackers.base_tracker.base_tracker import BaseObservation


@dataclass
class CharucoObservation(BaseObservation):
    all_charuco_ids: list[int]
    all_charuco_corners_in_object_coordinates: np.ndarray[..., 3]
    all_aruco_ids: list[int]
    all_aruco_corners_in_object_coordinates: np.ndarray[..., 3]

    detected_charuco_corner_ids: list[list[int]]|None
    detected_charuco_corners_image_coordinates: np.ndarray[..., 2]|None
    detected_charuco_corners_in_object_coordinates: np.ndarray[..., 3]|None


    detected_aruco_marker_ids: list[list[int]] | None
    detected_aruco_marker_corners: tuple[np.ndarray[..., 2]] | None

    translation_vector: np.ndarray[..., 3] | None
    rotation_vector: np.ndarray[..., 3] | None

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
            translation_vector=None,
            rotation_vector=None,
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

    def to_serializable_dict(self) -> dict:
        d =  {
            "all_charuco_ids": self.all_charuco_ids,
            "all_charuco_corners_in_object_coordinates": self.all_charuco_corners_in_object_coordinates.tolist(),
            "detected_charuco_corner_ids": self.detected_charuco_corner_ids.tolist() if self.detected_charuco_corner_ids is not None else None,
            "detected_charuco_corners_image_coordinates": self.detected_charuco_corners_image_coordinates.tolist() if self.detected_charuco_corners_image_coordinates is not None else None,
            "detected_charuco_corners_in_object_coordinates": self.detected_charuco_corners_in_object_coordinates.tolist() if self.detected_charuco_corners_in_object_coordinates is not None else None,
            "all_aruco_corners_in_object_coordinates": self.all_aruco_corners_in_object_coordinates.tolist(),
            "detected_aruco_marker_ids": self.detected_aruco_marker_ids.tolist() if self.detected_aruco_marker_ids is not None else None,
            "detected_aruco_marker_corners": [corner.tolist() for corner in self.detected_aruco_marker_corners] if self.detected_aruco_marker_corners is not None else None,
            "translation_vector": self.translation_vector.tolist() if self.translation_vector is not None else None,
            "rotation_vector": self.rotation_vector.tolist() if self.rotation_vector is not None else None,
            "image_size": self.image_size
        }
        try:
            json.dumps(d).encode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to serialize CharucoObservation to JSON: {e}")
        return d

    def to_json_string(self) -> str:
        return json.dumps(self.to_serializable_dict(), indent=4)

    def to_json_bytes(self) -> bytes:
        return self.to_json_string().encode("utf-8")

CharucoObservations = list[CharucoObservation]
