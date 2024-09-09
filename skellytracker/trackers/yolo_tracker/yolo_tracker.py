import numpy as np
from typing import Dict
from ultralytics import YOLO

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.yolo_tracker.yolo_model_info import YOLOModelInfo
from skellytracker.trackers.yolo_tracker.yolo_recorder import YOLORecorder


class YOLOPoseTracker(BaseTracker):
    def __init__(self, model_size: str = "nano"):
        super().__init__(tracked_object_names=[], recorder=YOLORecorder())

        pytorch_model = YOLOModelInfo.model_dictionary[model_size]
        self.model = YOLO(pytorch_model)

    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        # "max_det=1" argument to limit to single person tracking for now
        results = self.model(image, max_det=1, verbose=False)

        self.unpack_results(results)

        self.annotated_image = self.annotate_image(
            image=image, results=results, **kwargs
        )

        return self.tracked_objects

    def annotate_image(self, image: np.ndarray, results: list, **kwargs) -> np.ndarray:
        return results[-1].plot()

    def unpack_results(self, results: list):
        tracked_person = np.asarray(results[-1].keypoints.xy)
        self.tracked_objects["tracked_person"] = TrackedObject(
            object_id="tracked_person"
        )
        if tracked_person.size != 0:
            # add averages of all tracked points as pixel x and y
            self.tracked_objects["tracked_person"].pixel_x = float(
                np.mean(tracked_person[:, 0])
            )
            self.tracked_objects["tracked_person"].pixel_y = float(
                np.mean(tracked_person[:, 1])
            )
            self.tracked_objects["tracked_person"].extra["landmarks"] = tracked_person
        else:
            self.tracked_objects["tracked_person"].pixel_x = None
            self.tracked_objects["tracked_person"].pixel_y = None
            self.tracked_objects["tracked_person"].extra["landmarks"] = np.full(
                (1, YOLOModelInfo.num_tracked_points, 2), np.nan
            )


if __name__ == "__main__":
    YOLOPoseTracker().demo()
