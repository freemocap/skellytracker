import cv2
import numpy as np
from typing import Dict
from ultralytics import YOLO

from skelly_tracker.trackers.base_tracker.base_tracker import BaseTracker
from skelly_tracker.trackers.base_tracker.tracked_object import TrackedObject
from skelly_tracker.trackers.yolo_object_tracker.yolo_object_model_dictionary import (
    yolo_object_model_dictionary,
)
from skelly_tracker.trackers.yolo_object_tracker.yolo_object_recorder import YOLOObjectRecorder


class YOLOObjectTracker(BaseTracker):
    def __init__(self, model_size: str = "nano", person_only: bool = True):
        super().__init__(tracked_object_names=["object"], recorder=YOLOObjectRecorder())

        pytorch_model = yolo_object_model_dictionary[model_size]
        self.model = YOLO(pytorch_model)
        if person_only:
            self.classes = 0  # 0 is the YOLO class for person detection
        else:
            self.classes = None

    def process_image(self, image, **kwargs) -> Dict[str, TrackedObject]:
        results = self.model(image, classes=self.classes, max_det=1, verbose=False)

        box_xywh = np.asarray(results[0].boxes.xywh).flatten()

        if box_xywh.size > 0:
            self.tracked_objects["object"].pixel_x = box_xywh[0] + (box_xywh[2] * 0.5)
            self.tracked_objects["object"].pixel_y = box_xywh[1] + (box_xywh[3] * 0.5)

        self.tracked_objects["object"].extra["boxes_xywh"] = box_xywh
        self.tracked_objects["object"].extra["original_image_shape"] = results[0].boxes.orig_shape

        self.annotated_image = self.annotate_image(image, results=results, **kwargs)

        return self.tracked_objects

    def annotate_image(self, image: np.ndarray, results, **kwargs) -> np.ndarray:
        return results[0].plot()


if __name__ == "__main__":
    YOLOObjectTracker().demo()
