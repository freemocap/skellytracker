import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import cv2
import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, Field

from skellytracker.io.demo_viewers.image_demo_viewer import ImageDemoViewer
from skellytracker.io.demo_viewers.webcam_demo_viewer import WebcamDemoViewer
from skellytracker.trackers.base_tracker.point_cloud import PointCloud

logger = logging.getLogger(__name__)

TrackedPointIdString = str
TrackerTypeString = str

TrackedPoint2dArray = NDArray[Shape["2 xyz"], float]
TrackedPoints2dArray = NDArray[Shape["* number_of_points,2 xyz"], float]


class BaseObservation(ABC):
    """
    Base class for all tracker observations.

    Every observation carries a PointCloud as its canonical data. The PointCloud
    structurally couples point names with coordinate arrays — they cannot desync.

    Subclasses must be dataclasses (or any class) that provide:
        - points: PointCloud
        - frame_number: int
        - tracker_type: str
        - from_detection_results() classmethod

    The concrete methods (to_2d_array, to_tracked_points, etc.) all delegate
    to the PointCloud, so they are structurally guaranteed to be consistent.
    """

    # Subclasses must have these attributes
    points: PointCloud
    frame_number: int
    tracker_type: str

    @classmethod
    @abstractmethod
    def from_detection_results(cls, *args, **kwargs) -> "BaseObservation":
        ...

    # =========================================================================
    # Concrete methods — all delegate to PointCloud
    # =========================================================================

    def to_tracked_points(self, *, confidence_threshold: float | None = None) -> dict[TrackedPointIdString, TrackedPoint2dArray]:
        """
        Get all tracked points as {name: (x, y)} dict.

        When confidence_threshold is None, returns ALL points (including NaN)
        to maintain structural consistency with to_2d_array().
        """
        if confidence_threshold is not None:
            filtered = self.points.filtered_by_confidence(threshold=confidence_threshold)
            return filtered.to_named_dict(dimensions=2)
        return self.points.to_named_dict(dimensions=2)

    def to_2d_array(self, *, confidence_threshold: float | None = None, fill_with_nans: bool = True) -> TrackedPoints2dArray:
        """
        Convert observation to (N, 2) array.

        Always returns the same N rows in the same order as
        to_tracked_points().keys() — guaranteed by PointCloud.
        """
        if confidence_threshold is not None:
            filtered = self.points.filtered_by_confidence(
                threshold=confidence_threshold,
                fill_with_nans=fill_with_nans,
            )
            return filtered.to_2d_array()
        return self.points.to_2d_array()

    def get_confidence_scores(self) -> NDArray[Shape["* number_of_points"], float]:
        """Get visibility/confidence scores for all tracked points."""
        return self.points.visibility.copy()

    def filter_by_confidence(
            self,
            points: NDArray,
            confidence_scores: NDArray[Shape["* number_of_points"], float],
            confidence_threshold: float,
            fill_with_nans: bool = True,
    ) -> NDArray:
        """Filter a points array by confidence threshold."""
        if fill_with_nans:
            filtered_points = points.copy()
            mask = confidence_scores < confidence_threshold
            filtered_points[mask] = np.nan
            return filtered_points
        else:
            mask = confidence_scores >= confidence_threshold
            return points[mask]

    def to_json_string(self) -> str:
        """Serialize observation to JSON string."""
        data: dict = {
            "frame_number": self.frame_number,
            "tracker_type": self.tracker_type,
            "point_names": list(self.points.names),
            "xyz": self.points.xyz.tolist(),
            "visibility": self.points.visibility.tolist(),
        }
        return json.dumps(data, indent=4)

    def to_json_bytes(self) -> bytes:
        return self.to_json_string().encode("utf-8")


BaseObservations = list[BaseObservation]


class BaseImageAnnotatorConfig(BaseModel, ABC):
    show_overlay: bool = False


class BaseImageAnnotator(BaseModel, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )
    config: BaseImageAnnotatorConfig
    observations: BaseObservations  # make it a list to allow plotting trails, etc.

    @classmethod
    @abstractmethod
    def create(cls, config: BaseImageAnnotatorConfig):
        pass

    @abstractmethod
    def annotate_image(self, image: np.ndarray, observation: BaseObservation) -> np.ndarray:
        pass

    @staticmethod
    def draw_doubled_text(image: np.ndarray,
                          text: str,
                          x: int,
                          y: int,
                          font_scale: float,
                          color: tuple[int, int, int],
                          thickness: int):
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness * 3)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


class BaseDetectorConfig(BaseModel, ABC):
    confidence_threshold: float = Field(
        default=0.5,
        description="Default confidence threshold for filtering tracked points (0.0-1.0)"
    )


class BaseTrackerConfig(BaseModel, ABC):
    detector_config: BaseDetectorConfig
    annotator_config: BaseImageAnnotatorConfig | None = None


class BaseDetector(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: BaseDetectorConfig

    @classmethod
    def create(cls, config: BaseDetectorConfig):
        raise NotImplementedError("Must implement a method to create a detector from a config.")

    @abstractmethod
    def detect(self,
               frame_number: int,
               image: np.ndarray) -> BaseObservation:
        pass


class BaseRecorder(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    observations: List[BaseObservation] = Field(default_factory=list)

    def add_observation(self, observation: BaseObservation) -> None:
        self.observations.append(observation)

    @property
    def to_array(self) -> np.ndarray:
        return np.stack([observation.to_2d_array() for observation in self.observations])

    @property
    def to_json_string(self) -> str:
        output_dict = {
            frame_number: observation.to_json_string()
            for frame_number, observation in enumerate(self.observations)
        }
        return json.dumps(output_dict, indent=4)

    def save_array(self, output_path: Path) -> None:
        np.save(file=output_path, arr=self.to_array)

    def save_json_file(self, output_path: Path) -> None:
        with open(output_path, 'w') as json_file:
            json_file.write(self.to_json_string)

    def clear(self) -> None:
        self.observations = []


class BaseTracker(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: BaseTrackerConfig
    detector: BaseDetector
    annotator: BaseImageAnnotator
    recorder: BaseRecorder | None = None

    @classmethod
    def create(cls, config: BaseTrackerConfig):
        raise NotImplementedError("Must implement a method to create a tracker from a config.")

    def process_image(self,
                      frame_number: int,
                      image: np.ndarray,
                      record_observation: bool = True) -> BaseObservation:
        latest_observation = self.detector.detect(image=image, frame_number=frame_number)

        if record_observation and self.recorder is not None:
            self.recorder.add_observation(observation=latest_observation)

        return latest_observation

    def annotate_image(self, image: np.ndarray, observation: BaseObservation) -> np.ndarray:
        return self.annotator.annotate_image(image=image, observation=observation)

    def demo(self) -> None:
        camera_viewer = WebcamDemoViewer(
            tracker=self,
            window_title=self.__class__.__name__
        )
        camera_viewer.run()

    def image_demo(self, image_path: Path) -> None:
        image_viewer = ImageDemoViewer(self, self.__class__.__name__)
        image_viewer.run(image_path=image_path)
