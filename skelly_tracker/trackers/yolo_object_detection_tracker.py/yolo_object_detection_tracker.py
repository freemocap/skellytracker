import cv2
import numpy as np
from typing import Dict
from ultralytics import YOLO

from skelly_tracker.trackers.base_tracker.base_tracker import BaseTracker

class YOLOPoseTracker(BaseTracker):
    def __init__(self, model_size: str="nano"):
        super().__init__(tracked_object_names=["human_pose"])

        # pytorch_model = yolo_object_detection_model_dictionary[model_size]
        self.model = YOLO('yolov8n.pt')

    def process_image(self, image, **kwargs):
        results = self.model(image)

        self.tracked_objects["objects"].extra["landmarks"] = np.array(results[0].keypoints)

        self.annotated_image = self.annotate_image(image, results=results, **kwargs)

        return self.tracked_objects

    def annotate_image(self, image: np.ndarray, results, **kwargs) -> np.ndarray:
        return results[0].plot()
        

if __name__ == "__main__":
    YOLOPoseTracker().demo()
