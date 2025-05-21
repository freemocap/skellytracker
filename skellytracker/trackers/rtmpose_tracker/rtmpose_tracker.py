from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.rtmpose_tracker.rtmpose_model_info import RTMPoseModelInfo
from skellytracker.trackers.rtmpose_tracker.rtmpose_recorder import RTMPoseRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from rtmlib import Wholebody, draw_skeleton
import numpy as np
import platform
from typing import Dict
import logging



from pathlib import Path
import onnxruntime

logger = logging.getLogger(__name__)

class RTMPoseTracker(BaseTracker):
        def __init__(
            self,
        ):
                super().__init__(
                tracked_object_names=RTMPoseModelInfo.tracked_object_names,
                recorder = RTMPoseRecorder(),
                )

                device = self._get_device()
                
                self.wholebody_model = Wholebody(
                to_openpose = False,
                backend ='onnxruntime',
                device = device
                )

                if device == "cuda":
                        onnx_path = Path(self.wholebody_model.pose_model.onnx_model)
                        providers = ["CUDAExecutionProvider"] if device == device else ["CPUExecutionProvider"]
                        self.pose_session = onnxruntime.InferenceSession(str(onnx_path), providers=providers)
                f = 2

        def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
                slices = {"body_slice": slice(0, 23),
                        "face_slice": slice(23, 91),
                        "left_hand_slice": slice(91, 112),
                        "right_hand_slice": slice(112, 133)}

                keypoints, scores = self.wholebody_model(image)
                if keypoints.ndim == 3: #if multiple people are detected
                        results = np.array(keypoints[0])     # pick the primary person

                else:
                        results = keypoints
                self.tracked_objects["pose_landmarks"].extra["landmarks"] = results[slices["body_slice"]]
                self.tracked_objects["face_landmarks"].extra["landmarks"] = results[slices["face_slice"]]
                self.tracked_objects["left_hand_landmarks"].extra["landmarks"] = results[slices["left_hand_slice"]]
                self.tracked_objects["right_hand_landmarks"].extra["landmarks"] = results[slices["right_hand_slice"]]

                self.annotated_image = self.annotate_image(
                                        image=image,
                                        keypoints=keypoints,
                                        scores=scores)
                
        def annotate_image(self, image: np.ndarray, keypoints, scores) -> np.ndarray:
                image_annotated = draw_skeleton(image,
                                        keypoints,
                                        scores,
                                        openpose_skeleton=False,
                                        kpt_thr=0.5)
                return image_annotated
        
        def _get_device(self):
            if platform.system() == "Darwin":
                    device = "mps"
                    logger.debug("MacOS detected, attempting to use MPS")
            elif onnxruntime.get_device() == "GPU":
                    device = "cuda"
                    logger.debug("onnxruntime-gpu installed, attempting to use CUDA")
            else:
                    device = "cpu"
                    logger.debug("onnxruntime (non gpu version) installed, using CPU")
            return device

if __name__ == "__main__":
    RTMPoseTracker().demo()
