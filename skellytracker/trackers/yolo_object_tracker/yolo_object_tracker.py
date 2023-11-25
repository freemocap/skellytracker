import cv2
import numpy as np
from typing import Dict
from ultralytics import YOLO

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.yolo_object_tracker.yolo_object_model_dictionary import yolo_object_model_dictionary

class YOLOObjectTracker(BaseTracker):
    def __init__(self, model_size: str="nano"):
        super().__init__(tracked_object_names=["objects"])

        pytorch_model = yolo_object_model_dictionary[model_size]
        self.model = YOLO(pytorch_model)

    def process_image(self, image, **kwargs):
        results = self.model(image)

        self.tracked_objects["objects"].extra["landmarks"] = np.array(results[0].keypoints)

        self.annotated_image = self.annotate_image(image, results=results, **kwargs)

        return self.tracked_objects

    def annotate_image(self, image: np.ndarray, results, **kwargs) -> np.ndarray:
        return results[0].plot()
        

if __name__ == "__main__":
    YOLOObjectTracker().demo()
