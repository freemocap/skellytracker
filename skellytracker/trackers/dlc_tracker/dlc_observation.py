from typing import NamedTuple

import numpy as np
from numpydantic import NDArray, Shape

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation, TrackerTypeString, TrackedPoint2d

class DeepLabCutObservation(BaseObservation):
    tracker_type:TrackerTypeString = 'dlc_tracker'
    frame_number: int  # the frame number of the image in which this observation was made
    pose_points: np.ndarray
    confidence_values: np.ndarray
    image_size: tuple[int, int]

    @classmethod
    def from_detection_results(cls,
                               frame_number: int,
                               pose_prediction: dict[str, NDArray[Shape["1,N,3"], float]],
                               image_size: tuple[int, int]):
        # TODO: this will not work for multi animal dlc models
        prediction_values = pose_prediction["bodyparts"].squeeze()
        return cls(
            frame_number=frame_number,
            pose_points=prediction_values[:, :2],
            confidence_values=prediction_values[:, 2],
            image_size=image_size
        )

    def to_tracked_points(self) -> dict[str, TrackedPoint2d]:
        tracked_points_dict = {}
        for i in range(self.pose_points.shape[0]):
            tracked_points_dict[f"Point-{i}"] = self.pose_points[i, :2]
            tracked_points_dict[f"Point-{i}-Confidence"] = self.confidence_values[i]

        return tracked_points_dict

    def to_array(self) -> NDArray[Shape["N, 2"], float]:
        return self.pose_points
    

DLCObservations = list[DeepLabCutObservation]
