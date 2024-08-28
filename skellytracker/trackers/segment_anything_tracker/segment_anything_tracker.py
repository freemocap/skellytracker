import numpy as np
from ultralytics import SAM

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker


class SAMTracker(BaseTracker):
    def __init__(self):
        super().__init__(recorder=None, tracked_object_names=["segmentation"])

        self.model = SAM("sam_b.pt")

    def process_image(self, image, **kwargs):
        results = self.model.predict(image)

        self.tracked_objects["segmentation"].extra["landmarks"] = np.array(
            results[0].keypoints
        )

        self.annotated_image = self.annotate_image(image, results=results, **kwargs)

        return self.tracked_objects

    def annotate_image(self, image: np.ndarray, results, **kwargs) -> np.ndarray:
        return results[0].plot()


if __name__ == "__main__":
    SAMTracker().demo()
