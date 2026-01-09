import numpy as np
from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation, TrackerTypeString

class VITPoseObservation(BaseObservation):
    tracker_type: TrackerTypeString = "vitpose"
    frame_number: int
    image_size: tuple[int, int]  
    keypoints: np.ndarray

    @classmethod
    def from_detection_results(cls, frame_number: int, results: dict[str, np.ndarray], image_size: tuple[int, int]):

        results_array = results[0]  #only picking the first 'person' from the data

        return cls(
            frame_number=frame_number,
            keypoints=results_array,
            image_size=image_size
        )
    
    def to_2d_array(self, *, confidence_threshold:float|None = None, fill_with_nans: bool = True) -> np.ndarray:
        point_2d = self.keypoints[...,:2]

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
 