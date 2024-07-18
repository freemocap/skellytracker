import numpy as np
from typing import Dict
from ultralytics import YOLO

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.yolo_tracker.yolo_model_info import YOLOModelInfo
from skellytracker.trackers.yolo_tracker.yolo_recorder import YOLORecorder

class YOLOPoseTracker(BaseTracker):
    def __init__(self, model_size: str = "nano", max_tracked_objects: int = 1):
        self.tracked_object_names = ["tracked_person_" + str(i) for i in range(max_tracked_objects)]
        super().__init__(tracked_object_names=self.tracked_object_names, recorder=YOLORecorder())

        pytorch_model = YOLOModelInfo.model_dictionary[model_size]
        self.model = YOLO(pytorch_model)

        self.max_tracked_objects = max_tracked_objects

    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        results = self.model(image, max_det=self.max_tracked_objects, verbose=False)

        self.unpack_results(results)

        self.annotated_image = self.annotate_image(image=image, results=results, **kwargs)

        return self.tracked_objects

    def annotate_image(self, image: np.ndarray, results: list, **kwargs) -> np.ndarray:
        return results[-1].plot()

    def unpack_results(self, results: list):
        tracked_person = np.asarray(results[-1].keypoints.xy)

        if tracked_person.size == 0:
            # reinitialize tracked objects
            for i in range(self.max_tracked_objects):
                self.tracked_objects[f"tracked_person_{i}"] = TrackedObject(
                    object_id=f"tracked_person_{i}"
                )
                self.tracked_objects[f"tracked_person_{i}"].extra["landmarks"] = np.full(
                    (1, YOLOModelInfo.num_tracked_points, 2), np.nan
                )

        for i in range(tracked_person.shape[0]):
            self.tracked_objects[f"tracked_person_{i}"] = TrackedObject(
                object_id=f"tracked_person_{i}"
            )
            # add averages of all tracked points as pixel x and y
            self.tracked_objects[f"tracked_person_{i}"].pixel_x = np.mean(tracked_person[i, :, 0])
            self.tracked_objects[f"tracked_person_{i}"].pixel_y = np.mean(tracked_person[i, :, 1])
            self.tracked_objects[f"tracked_person_{i}"].extra["landmarks"] = tracked_person[i, :, :]

        for i in range(tracked_person.shape[0], self.max_tracked_objects):
            # reinitialize tracked objects that weren't filled
            self.tracked_objects[f"tracked_person_{i}"] = TrackedObject(
                object_id=f"tracked_person_{i}"
            )
            self.tracked_objects[f"tracked_person_{i}"].extra["landmarks"] = np.full(
                (1, YOLOModelInfo.num_tracked_points, 2), np.nan
            )

        


if __name__ == "__main__":
    # YOLOPoseTracker().demo()
    from pathlib import Path
    YOLOPoseTracker(max_tracked_objects=10).image_demo(Path("/Users/philipqueen/Downloads/bus.jpg"))