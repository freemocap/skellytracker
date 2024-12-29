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
class CharucoDetection:
    """
    Output of the CharucoTracker.detect() method, to be re-shaped into a CharucoObservation.
    """
    charuco_corners: np.ndarray
    charuco_corner_ids: list[list[int]]
    aruco_marker_corners: np.ndarray
    aruco_marker_ids: list[list[int]]

@dataclass
class CharucoObservationFactory(BaseObservationFactory):
    charuco_corner_ids: List[int]
    aruco_marker_ids: List[int]

    def create_observation(self,
                           detection: CharucoDetection) -> CharucoObservation:


        charuco_corners_out = {id: None for id in self.charuco_corner_ids}
        aruco_marker_corners_out = {id: None for id in self.aruco_marker_ids}

        if detection.charuco_corner_ids is None or len(detection.charuco_corner_ids) == 0:
            detection.charuco_corner_ids = None
            detection.charuco_corners = None
        elif len(detection.charuco_corner_ids) == 1:
            detection.charuco_corner_ids = list(detection.charuco_corner_ids[0])
            detection.charuco_corners = list(detection.charuco_corners[0])
        elif len(detection.charuco_corner_ids) > 1:
            detection.charuco_corner_ids = list(np.squeeze(detection.charuco_corner_ids))
            detection.charuco_corners = list(np.squeeze(detection.charuco_corners))

        if detection.aruco_marker_ids is None or len(detection.aruco_marker_ids) == 0:
            detection.aruco_marker_ids = None
            detection.aruco_marker_corners = None
        elif len(detection.aruco_marker_ids) == 1:
            detection.aruco_marker_ids = list(detection.aruco_marker_ids[0])
            detection.aruco_marker_corners = list(detection.aruco_marker_corners[0])
        elif len(detection.aruco_marker_ids) > 1:
            detection.aruco_marker_ids = list(np.squeeze(detection.aruco_marker_ids))
            detection.aruco_marker_corners = list(np.squeeze(detection.aruco_marker_corners))

        if detection.charuco_corner_ids is not None:
            for corner_id, corner in zip(detection.charuco_corner_ids, detection.charuco_corners):
                charuco_corners_out[corner_id] = corner

        if detection.aruco_marker_ids is not None:
            for marker_id, corner in zip(detection.aruco_marker_ids, detection.aruco_marker_corners):
                aruco_marker_corners_out[marker_id] = corner

        return CharucoObservation(
            charuco_corners=charuco_corners_out,
            aruco_marker_corners=aruco_marker_corners_out,
        )
