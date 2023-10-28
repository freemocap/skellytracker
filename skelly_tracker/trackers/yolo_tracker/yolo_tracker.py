import numpy as np
from typing import Dict
from ultralytics import YOLO

from skelly_tracker.trackers.base_tracker.base_tracker import BaseTracker, TrackedObject
from skelly_tracker.trackers.yolo_tracker.yolo_model_dictionary import (
    yolo_model_dictionary,
)


class YOLOPoseTracker(BaseTracker):
    def __init__(self, model_size: str = "nano"):
        super().__init__(tracked_object_names=[])

        pytorch_model = yolo_model_dictionary[model_size]
        self.model = YOLO(pytorch_model)

    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        results = self.model(image)
        
        self.unpack_results(results)

        self.annotated_image = self.annotate_image(image, results=results, **kwargs)

        return self.tracked_objects

    def annotate_image(self, image: np.ndarray, results, **kwargs) -> np.ndarray:
        return results[-1].plot()

    def unpack_results(self, results):
        tracked_person_number = 0

        for tracked_person in np.asarray(results[-1].keypoints.xy):
            tracked_person_name = f"tracked_person_{tracked_person_number}"

            self.tracked_objects[tracked_person_name] = TrackedObject(
                object_id=tracked_person_name
            )
            # add averages of all tracked points as pixel x and y
            self.tracked_objects[tracked_person_name].pixel_x = np.mean(
                tracked_person[:, 0], axis=0
            )
            self.tracked_objects[tracked_person_name].pixel_y = np.mean(
                tracked_person[:, 1], axis=0
            )
            self.tracked_objects[tracked_person_name].extra[
                "landmarks"
            ] = tracked_person

            tracked_person_number += 1


if __name__ == "__main__":
    YOLOPoseTracker().demo()
