from skellytracker.trackers.base_tracker.base_recorder import BaseRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.rtmpose_tracker.rtmpose_model_info import RTMPoseModelInfo
from typing import Dict
import numpy as np

class RTMPoseRecorder(BaseRecorder):
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        self.recorded_objects.append(
            [
                tracked_objects[tracked_object_name]
                for tracked_object_name in tracked_objects.keys()
            ]
        )

    def process_tracked_objects(self, **kwargs) -> np.ndarray:
        num_frames = len(self.recorded_objects)
        num_keypoints = RTMPoseModelInfo.num_tracked_points

        self.recorded_objects_array = np.full(
            (num_frames, num_keypoints, 3), np.nan
        )

        for i, recorded_object_list in enumerate(self.recorded_objects):
            tracked_obj = recorded_object_list[0]  # we assume one person per frame
            keypoints = tracked_obj.extra.get("landmarks", None)  # shape: (133, 2)
            scores = tracked_obj.extra.get("scores", None)        # shape: (133,)

            if keypoints is not None:
                self.recorded_objects_array[i, :, 0:2] = keypoints  # X, Y
                self.recorded_objects_array[i, :, 2] = scores       # Z holds score for now

        return self.recorded_objects_array