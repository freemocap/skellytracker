import numpy as np
from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation, TrackerTypeString

class VITPoseObservation(BaseObservation):
    tracker_type: TrackerTypeString = "vitpose"
    frame_number: int
    image_size: tuple[int, int]  
    keypoints: np.ndarray

    @classmethod
    def from_detection_results(cls, frame_number: int, results: dict[str, np.ndarray], image_size: tuple[int, int]):

        # Handle no detections
        if len(results) == 0:
            # Return NaN-filled array - 133 keypoints for wholebody (this number is unlikely to change, unless we also allow for other kinds of VIT models. At that point we'll need to account for that)
            num_keypoints = 133
            keypoints = np.full((num_keypoints, 3), np.nan, dtype=np.float32)
        else:
            keypoints = results[0]  # First person only

        return cls(
            frame_number=frame_number,
            keypoints=keypoints,
            image_size=image_size
        )
    
    def to_2d_array(self, *, confidence_threshold:float|None = None, fill_with_nans: bool = True) -> np.ndarray:
        point_2d = self.keypoints[..., :2][:, [1, 0]]  # Convert (y, x) to (x, y)

        if confidence_threshold is not None:
            confidence_scores = self.keypoints[...,2]

            point_2d = self.filter_by_confidence(
                points=point_2d,
                confidence_scores=confidence_scores,
                fill_with_nans=fill_with_nans   
            )

        return point_2d
    
    def to_tracked_points(self):
        pass
 
