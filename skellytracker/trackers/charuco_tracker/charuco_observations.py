from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from skellytracker.trackers.base_tracker.base_tracker import BaseObservation, BaseObservationFactory


@dataclass
class CharucoObservation(BaseObservation):
    charuco_corners_image: dict[int, tuple[float, float] | None]
    aruco_marker_corners: dict[int, list[tuple[float, float]] | None]
    charuco_corners_object: List[np.ndarray]
    image_size: Tuple[int, ...]

CharucoObservations = list[CharucoObservation]


@dataclass
class CharucoObservationFactory(BaseObservationFactory):
    charuco_corner_ids: List[int]
    aruco_marker_ids: List[int]
    charuco_corner_object_coordinates: List[np.ndarray]

    def create_observation(self,
                           detected_charuco_corners: np.ndarray,
                           detected_charuco_corner_ids: list[list[int]],
                           detected_aruco_marker_corners: np.ndarray,
                           detected_aruco_marker_ids: list[list[int]],
                           image_size: Tuple[int, ...]
                           ) -> CharucoObservation:

        charuco_corners_out = {id: None for id in self.charuco_corner_ids}
        aruco_marker_corners_out = {id: None for id in self.aruco_marker_ids}

        (formatted_aruco_corners,
         formatted_aruco_ids,
         formatted_charuco_ids,
         formatted_charuco_corners) = self._format_input_data(
             detected_aruco_marker_corners,
             detected_aruco_marker_ids,
             detected_charuco_corner_ids,
             detected_charuco_corners
         )

        if formatted_charuco_ids:
            for corner_id, corner in zip(formatted_charuco_ids, formatted_charuco_corners):
                charuco_corners_out[corner_id] = corner

        if formatted_aruco_ids:
            for marker_id, corner in zip(formatted_aruco_ids, formatted_aruco_corners):
                aruco_marker_corners_out[marker_id] = corner

        return CharucoObservation(
            charuco_corners_image=charuco_corners_out,
            aruco_marker_corners=aruco_marker_corners_out,
            charuco_corners_object=self.charuco_corner_object_coordinates,
            image_size=image_size
        )

    def _format_input_data(self, detected_aruco_marker_corners, detected_aruco_marker_ids,
                           detected_charuco_corner_ids, detected_charuco_corners):
        def format_data(ids, corners):
            if ids is None or len(ids) == 0:
                return None, None
            if len(ids) == 1:
                return list(ids[0]), list(corners[0])
            return list(np.squeeze(ids)), list(np.squeeze(corners))

        formatted_charuco_ids, formatted_charuco_corners = format_data(detected_charuco_corner_ids, detected_charuco_corners)
        formatted_aruco_ids, formatted_aruco_corners = format_data(detected_aruco_marker_ids, detected_aruco_marker_corners)

        return formatted_aruco_corners, formatted_aruco_ids, formatted_charuco_ids, formatted_charuco_corners