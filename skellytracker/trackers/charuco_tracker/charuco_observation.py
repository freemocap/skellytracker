import json

from numpy.random.mtrand import Sequence
from numpydantic import NDArray, Shape

import numpy as np

from skellytracker.trackers.base_tracker.base_tracker import BaseObservation


class CharucoObservation(BaseObservation):
    all_charuco_ids: list[int]
    all_charuco_corners_in_object_coordinates: NDArray[Shape["* charuco_id, 3 xyz"], np.float32]

    all_aruco_ids: list[int]
    all_aruco_corners_in_object_coordinates: NDArray[Shape["* aruco_id, 4 corners, 3 xyz"], np.float32]

    detected_charuco_corner_ids: NDArray[Shape["*, ..."], int]|None
    detected_charuco_corners_image_coordinates: NDArray[Shape[" * charuco_id, 2 pxpy"], np.float32]|None
    detected_charuco_corners_in_object_coordinates: NDArray[Shape["* charuco_id, 3 xyz"], np.float32]|None


    detected_aruco_marker_ids: NDArray[Shape["* , ..."], np.int32] | None
    detected_aruco_marker_corners: Sequence[NDArray[Shape["4 corners, 2 pxpy"], np.float32]] | None

    translation_vector: NDArray[Shape["3 tx_ty_tz"], np.float32] | None
    rotation_vector: NDArray[Shape["3 rx_ry_rz"], np.float32] | None

    image_size: tuple[int, int]

    @classmethod
    def from_detect_board_results(cls,
                                  detected_charuco_corners: NDArray[Shape["* charuco_id,  2 pxpy"], float],
                                  detected_charuco_corner_ids: NDArray[Shape["* charuco_id, ..."], int],
                                  detected_aruco_marker_corners: Sequence[NDArray[Shape["*, 2"], float]],
                                  detected_aruco_marker_ids: NDArray[Shape["* aruco_id, ..."], int],
                                  all_charuco_ids: list[int],
                                  all_charuco_corners_in_object_coordinates: NDArray[Shape["* charuco_id, 3 xyz"], float],
                                  all_aruco_ids: list[int],
                                  all_aruco_corners_in_object_coordinates: [NDArray[Shape["* aruco_id, 4 corners, 3 xyz"], float]],
                                  image_size: tuple[int, int]):


        if detected_aruco_marker_ids is not None:
            # squeeze out singleton dimensions (i.e. a.shape = [2,1,3] -> np.squeeze(a).shape = [2,3])

            if detected_aruco_marker_ids.shape == (1, 1):
                # deal with special case where only one marker is detected
                detected_aruco_marker_ids = detected_aruco_marker_ids[0]
            else:
                detected_aruco_marker_ids = np.squeeze(detected_aruco_marker_ids)
            detected_aruco_marker_corners = tuple([np.squeeze(corner) for corner in detected_aruco_marker_corners])

        detected_charuco_corners_in_object_coordinates: NDArray[Shape["* charuco_id, 3 xyz"], np.float32] | None = None
        if detected_charuco_corner_ids is not None:
            if detected_charuco_corner_ids.shape == (1, 1):
                detected_charuco_corner_ids = detected_charuco_corner_ids[0]
                detected_charuco_corners = detected_charuco_corners[0]
            else:
                detected_charuco_corner_ids = np.squeeze(detected_charuco_corner_ids)
                detected_charuco_corners = np.squeeze(detected_charuco_corners)

            detected_charuco_corners_in_object_coordinates = all_charuco_corners_in_object_coordinates[detected_charuco_corner_ids, :]

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
            corner_dict[corner_id] = np.squeeze(self.detected_charuco_corners_image_coordinates[corner_index])
        return corner_dict

    @property
    def aruco_corners_dict(self) -> dict[int, np.ndarray[4, 2]]:
        corner_dict = {}
        if self.aruco_empty:
            return corner_dict
        for corner_index, corner_id in enumerate(self.detected_aruco_marker_ids):
            corner_dict[corner_id] = np.squeeze(self.detected_aruco_marker_corners[corner_index])
        return corner_dict

    # def to_serializable_dict(self) -> dict:
    #     d =  {
    #         "all_charuco_ids": self.all_charuco_ids,
    #         "all_charuco_corners_in_object_coordinates": self.all_charuco_corners_in_object_coordinates.tolist(),
    #         "detected_charuco_corner_ids": self.detected_charuco_corner_ids.tolist() if self.detected_charuco_corner_ids is not None else None,
    #         "detected_charuco_corners_image_coordinates": self.detected_charuco_corners_image_coordinates.tolist() if self.detected_charuco_corners_image_coordinates is not None else None,
    #         "detected_charuco_corners_in_object_coordinates": self.detected_charuco_corners_in_object_coordinates.tolist() if self.detected_charuco_corners_in_object_coordinates is not None else None,
    #         "all_aruco_corners_in_object_coordinates": self.all_aruco_corners_in_object_coordinates.tolist(),
    #         "detected_aruco_marker_ids": self.detected_aruco_marker_ids.tolist() if self.detected_aruco_marker_ids is not None else None,
    #         "detected_aruco_marker_corners": [corner.tolist() for corner in self.detected_aruco_marker_corners] if self.detected_aruco_marker_corners is not None else None,
    #         "translation_vector": self.translation_vector.tolist() if self.translation_vector is not None else None,
    #         "rotation_vector": self.rotation_vector.tolist() if self.rotation_vector is not None else None,
    #         "image_size": self.image_size
    #     }
    #     try:
    #         json.dumps(d).encode("utf-8")
    #     except Exception as e:
    #         raise ValueError(f"Failed to serialize CharucoObservation to JSON: {e}")
    #     return d
    #
    # def to_json_string(self) -> str:
    #     return json.dumps(self.to_serializable_dict(), indent=4)
    #
    # def to_json_bytes(self) -> bytes:
    #     return self.to_json_string().encode("utf-8")

CharucoObservations = list[CharucoObservation]
