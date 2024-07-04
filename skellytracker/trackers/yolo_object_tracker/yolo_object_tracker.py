import numpy as np
from typing import Dict
from ultralytics import YOLO

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.yolo_object_tracker.yolo_object_model_info import (
    yolo_object_model_dictionary,
)
from skellytracker.trackers.yolo_object_tracker.yolo_object_recorder import (
    YOLOObjectRecorder,
)


class YOLOObjectTracker(BaseTracker):
    def __init__(
        self,
        model_size: str = "nano",
        person_only: bool = True,
        confidence_threshold: float = 0.5,
    ):
        super().__init__(tracked_object_names=["object"], recorder=YOLOObjectRecorder())

        pytorch_model = yolo_object_model_dictionary[model_size]
        self.model = YOLO(pytorch_model)
        self.confidence_threshold = confidence_threshold
        # TODO: When we expose this in freemocap, replace this with an int/list[int] to decide which class to track
        # TODO: Will also need to parameterize the "max_det" and setup tracker to take multiple tracked objects
        if person_only:
            self.classes = 0  # 0 is the YOLO class for person detection
        else:
            self.classes = None  # None includes all classes

    def process_image(self, image, **kwargs) -> Dict[str, TrackedObject]:
        results = self.model(
            image,
            classes=self.classes,
            max_det=1,
            verbose=False,
            conf=self.confidence_threshold,
        )

        box_xyxy = np.asarray(
            results[0].boxes.xyxy.cpu()
        ).flatten()  # On GPU, need to copy to CPU before np array conversion

        if box_xyxy.size > 0:
            self.tracked_objects["object"].pixel_x = (box_xyxy[0] + box_xyxy[2]) / 0.5
            self.tracked_objects["object"].pixel_y = (box_xyxy[1] + box_xyxy[3]) / 0.5

        self.tracked_objects["object"].extra["boxes_xyxy"] = box_xyxy
        self.tracked_objects["object"].extra["original_image_shape"] = results[
            0
        ].boxes.orig_shape

        self.annotated_image = self.annotate_image(image, results=results, **kwargs)

        return self.tracked_objects

    def annotate_image(self, image: np.ndarray, results, **kwargs) -> np.ndarray:
        return results[0].plot()


if __name__ == "__main__":
    YOLOObjectTracker().demo()
