import cv2
import numpy as np
from typing import Dict
from ultralytics import YOLO
from skelly_tracker.trackers.base_tracker.base_tracker import BaseTracker, TrackedObject

class YOLOPoseTracker(BaseTracker):
    def __init__(self):
        super().__init__(tracked_object_names=["human_pose"])

        self.model = YOLO('yolov8n-pose.pt')

    def process_image(self, image, **kwargs):
        self.results = self.model(image)

        self.tracked_objects["human_pose"].extra["landmarks"] = np.array(self.results[0].keypoints)
        print(self.tracked_objects["human_pose"])
        self.annotated_image = self.annotate_image(image, tracked_objects=self.tracked_objects)

        return self.tracked_objects

    def annotate_image(self, image: np.ndarray, tracked_objects: Dict[str, TrackedObject], **kwargs) -> np.ndarray:
        return self.results[0].plot()
        

if __name__ == "__main__":
    YOLOPoseTracker().demo()
