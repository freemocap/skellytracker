import numpy as np
from typing import Dict
from ultralytics import YOLO

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.yolo_tracker.yolo_model_info import YOLOModelInfo
from skellytracker.trackers.yolo_tracker.yolo_recorder import YOLORecorder

class YOLOPoseTracker(BaseTracker):
    def __init__(self, model_size: str = "nano", max_det: int = 1):
        self.tracked_object_names = ["tracked_person_" + str(i) for i in range(max_det)]
        super().__init__(tracked_object_names=self.tracked_object_names, recorder=YOLORecorder())

        pytorch_model = YOLOModelInfo.model_dictionary[model_size]
        self.model = YOLO(pytorch_model)

        self.max_det = max_det

    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        # "max_det=1" argument to limit to single person tracking for now
        results = self.model(image, max_det=self.max_det, verbose=False)

        self.unpack_results(results)
        print(f"YOLO results: {results[-1].keypoints}")

        self.annotated_image = self.annotate_image(image=image, results=results, **kwargs)

        return self.tracked_objects

    def annotate_image(self, image: np.ndarray, results: list, **kwargs) -> np.ndarray:
        return results[-1].plot()

    def unpack_results(self, results: list):
        tracked_person = np.asarray(results[-1].keypoints.xy)

        if tracked_person.size == 0:
            # reinitialize tracked objects
            for i in range(self.max_det):
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
            self.tracked_objects[f"tracked_person_{i}"].pixel_x = np.mean(tracked_person[:, 0])
            self.tracked_objects[f"tracked_person_{i}"].pixel_y = np.mean(tracked_person[:, 1])
            self.tracked_objects[f"tracked_person_{i}"].extra["landmarks"] = tracked_person

        for i in range(tracked_person.shape[0], self.max_det):
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
    YOLOPoseTracker(max_det=10).image_demo(Path("/Users/philipqueen/Downloads/bus.jpg"))