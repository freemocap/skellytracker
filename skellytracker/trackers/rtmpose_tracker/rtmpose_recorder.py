from skellytracker.trackers.base_tracker.base_recorder import BaseRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.rtmpose_tracker.rtmpose_model_info import RTMPoseModelInfo
from typing import Dict
import numpy as np
from copy import deepcopy
class RTMPoseRecorder(BaseRecorder):
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        self.recorded_objects.append(
            [
                deepcopy(tracked_objects[tracked_object_name])
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

            # one big (133, 2) buffer for this frame
            frame_xy   = np.full((RTMPoseModelInfo.num_tracked_points, 2), np.nan)
            frame_conf = np.full((RTMPoseModelInfo.num_tracked_points,),   np.nan)

            for obj in recorded_object_list:
                if obj.extra.get("landmarks") is None:
                    continue

                # choose the slice for this object_id
                if   obj.object_id == "pose_landmarks":        sl = slice(0, 23)
                elif obj.object_id == "face_landmarks":        sl = slice(23, 91)
                elif obj.object_id == "left_hand_landmarks":   sl = slice(91, 112)
                elif obj.object_id == "right_hand_landmarks":  sl = slice(112,133)
                else:
                    continue  # unexpected id → skip

                kps    = np.asarray(obj.extra["landmarks"])        # (n,2)
                scores = np.asarray(obj.extra.get("scores", []))   # (n,)

                # pad / truncate to the exact slice length
                n = min(kps.shape[0], sl.stop - sl.start)
                frame_xy[sl.start : sl.start+n, :] = kps[:n]
                if scores.size:
                    frame_conf[sl.start : sl.start+n] = scores[:n]

            # write into the master array
            self.recorded_objects_array[i, :, 0:2] = frame_xy
            self.recorded_objects_array[i, :, 2]   = frame_conf
            
        return self.recorded_objects_array