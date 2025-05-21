import cv2
import numpy as np
from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetector, BaseDetectorConfig
from skellytracker.trackers.bright_point_tracker.brightest_point_observation import BrightPatch, BrightestPointObservation



class BrightestPointDetectorConfig(BaseDetectorConfig):
    num_tracked_points: int = 1
    luminance_threshold: int = 200

class BrightestPointDetector(BaseDetector):
    config: BrightestPointDetectorConfig

    @classmethod
    def create(cls, config: BrightestPointDetectorConfig):
        return cls(config=config)

    def detect(self,
               frame_number: int,
               image: np.ndarray) -> BrightestPointObservation:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to get only bright regions
        _, thresholded_image = cv2.threshold(
            gray_image, self.config.luminance_threshold, 255, cv2.THRESH_BINARY
        )

        # Find contours of the bright regions
        bright_patches, _ = cv2.findContours(
            thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Process each bright patch separately
        patch_list = []
        for patch in bright_patches:
            patch_moments = cv2.moments(patch)
            if patch_moments["m00"] != 0:  # Avoid division by zero
                centroid_x = int(patch_moments["m10"] / patch_moments["m00"])
                centroid_y = int(patch_moments["m01"] / patch_moments["m00"])

                patch_list.append(
                    BrightPatch(
                        area=cv2.contourArea(patch),
                        centroid_x=centroid_x,
                        centroid_y=centroid_y,
                    )
                ) 

        largest_patches = sorted(
            patch_list, key=lambda patch: patch.area, reverse=True
        )[: min(len(patch_list), self.config.num_tracked_points)]

        if len(largest_patches) < self.config.num_tracked_points:
            largest_patches = largest_patches + [None] * (self.config.num_tracked_points - len(largest_patches))

        return BrightestPointObservation.from_detection_results(frame_number=frame_number, bright_patches=largest_patches)