from collections import deque
from dataclasses import dataclass
from typing import List

import numpy as np

from skellytracker.trackers.base_tracker.base_tracker import BaseObservation, BaseObservationFactory


@dataclass
class CharucoObservation(BaseObservation):
    charuco_corners: dict[int, tuple[float, float] | None]
    aruco_marker_corners: dict[int, list[tuple[float, float]] | None]


CharucoObservations = list[CharucoObservation]


@dataclass
class CharucoObservationFactory(BaseObservationFactory):
    charuco_corner_ids: List[int]
    aruco_marker_ids: List[int]

    def create_observation(self,
                           detected_charuco_corners: np.ndarray,
                           detected_charuco_corner_ids: list[list[int]],
                           detected_aruco_marker_corners: np.ndarray,
                           detected_aruco_marker_ids: list[list[int]],
                           ) -> CharucoObservation:
        charuco_corners_out = {id: None for id in self.charuco_corner_ids}
        aruco_marker_corners_out = {id: None for id in self.aruco_marker_ids}

        if detected_charuco_corner_ids is None or len(detected_charuco_corner_ids) == 0:
            detected_charuco_corner_ids = None
            detected_charuco_corners = None
        elif len(detected_charuco_corner_ids) == 1:
            detected_charuco_corner_ids = list(detected_charuco_corner_ids[0])
            detected_charuco_corners = list(detected_charuco_corners[0])
        elif len(detected_charuco_corner_ids) > 1:
            detected_charuco_corner_ids = list(np.squeeze(detected_charuco_corner_ids))
            detected_charuco_corners = list(np.squeeze(detected_charuco_corners))

        if detected_aruco_marker_ids is None or len(detected_aruco_marker_ids) == 0:
            detected_aruco_marker_ids = None
            detected_aruco_marker_corners = None
        elif len(detected_aruco_marker_ids) == 1:
            detected_aruco_marker_ids = list(detected_aruco_marker_ids[0])
            detected_aruco_marker_corners = list(detected_aruco_marker_corners[0])
        elif len(detected_aruco_marker_ids) > 1:
            detected_aruco_marker_ids = list(np.squeeze(detected_aruco_marker_ids))
            detected_aruco_marker_corners = list(np.squeeze(detected_aruco_marker_corners))

        if detected_charuco_corner_ids is not None:
            for corner_id, corner in zip(detected_charuco_corner_ids, detected_charuco_corners):
                charuco_corners_out[corner_id] = corner

        if detected_aruco_marker_ids is not None:
            for marker_id, corner in zip(detected_aruco_marker_ids, detected_aruco_marker_corners):
                aruco_marker_corners_out[marker_id] = corner

        return CharucoObservation(
            charuco_corners=charuco_corners_out,
            aruco_marker_corners=aruco_marker_corners_out,
        )
