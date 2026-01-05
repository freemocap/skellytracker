import numpy as np
from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation, TrackerTypeString
from skellytracker.trackers.rtmpose_tracker.rtmpose_landmark_names import BODY_LANDMARK_NAMES, FACE_LANDMARK_NAMES, LEFT_HAND_LANDMARK_NAMES, RIGHT_HAND_LANDMARK_NAMES

class RTMPoseObservation(BaseObservation):
    tracker_type: TrackerTypeString = "rtmpose"
    frame_number: int
    image_size: tuple[int, int]
    keypoints: np.ndarray
    scores: np.ndarray

    @classmethod
    def from_detection_results(cls, frame_number: int, keypoints: np.ndarray, score: np.ndarray, image_size: tuple[int, int]):
        return cls(frame_number=frame_number, image_size=image_size, keypoints=keypoints, scores=score)

    def to_2d_array(self, *, confidence_threshold: float | None = None, fill_with_nans: bool = True) -> np.ndarray:

        point_2d = self.keypoints[0,:] #for now, choosing 2d points for the first 'person' detected

        if confidence_threshold is not None:
            confidence_scores = self.scores[0,:] #for now, choosing scores for the first 'person' detected

            point_2d = self.filter_by_confidence(
                points = point_2d,
                confidence_scores = confidence_scores,
                confidence_threshold=confidence_threshold,
                fill_with_nans=fill_with_nans
            )
