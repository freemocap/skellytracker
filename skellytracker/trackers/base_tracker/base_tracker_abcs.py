import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


from skellytracker.trackers.base_tracker.point_cloud import PointCloud

logger = logging.getLogger(__name__)

TrackedPointIdString = str
TrackerTypeString = str


class TrackerType(str, Enum):
    CHARUCO = "charuco"
    MEDIAPIPE = "mediapipe" # composite of body, hands, and face - like holistic
    MEDIAPIPE_POSE = "mediapipe_pose"
    MEDIAPIPE_FACE = "mediapipe_face"
    MEDIAPIPE_HAND = "mediapipe_hand"
    LEGACY_MEDIAPIPE = "legacy_mediapipe"
    RTMPOSE = "rtmpose"

# Shape-annotated aliases kept for documentation; numpy.typing.NDArray is used
# in method signatures because it is beartype-compatible.
TrackedPoint2dArray = NDArray[np.float64]       # shape (2,)  — (x, y)
TrackedPoints2dArray = NDArray[np.float64]      # shape (N, 2) — one row per point


class BaseObservation(ABC):
    """
    Base class for all tracker observations.

    Every observation carries a PointCloud as its canonical data. The PointCloud
    structurally couples point names with coordinate arrays — they cannot desync.

    Subclasses must be dataclasses (or any class) that provide:
        - points: PointCloud
        - frame_number: int
        - tracker_type: TrackerType
        - from_detection_results() classmethod

    The concrete methods (to_2d_array, to_tracked_points, etc.) all delegate
    to the PointCloud, so they are structurally guaranteed to be consistent.
    """

    # Subclasses must have these attributes
    points: PointCloud
    frame_number: int
    tracker_type: TrackerType

    @classmethod
    @abstractmethod
    def from_detection_results(cls, *args, **kwargs) -> "BaseObservation":
        ...

    # =========================================================================
    # Concrete methods — all delegate to PointCloud
    # =========================================================================

    def to_tracked_points(
        self,
        *,
        confidence_threshold: float | None = None,
    ) -> dict[TrackedPointIdString, NDArray[np.float64]]:
        """
        Get all tracked points as {name: (x, y)} dict.

        When confidence_threshold is None, returns ALL points (including NaN)
        to maintain structural consistency with to_2d_array().
        """
        if confidence_threshold is not None:
            filtered = self.points.filtered_by_confidence(threshold=confidence_threshold)
            return filtered.to_named_dict(dimensions=2)
        return self.points.to_named_dict(dimensions=2)

    def to_2d_array(
        self,
        *,
        confidence_threshold: float | None = None,
        fill_with_nans: bool = True,
    ) -> NDArray[np.float64]:
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

    def get_confidence_scores(self) -> NDArray[np.float64]:
        """Get visibility/confidence scores for all tracked points."""
        return self.points.visibility.copy()

    def filter_by_confidence(
        self,
        points: NDArray[np.float64],
        confidence_scores: NDArray[np.float64],
        confidence_threshold: float,
        fill_with_nans: bool = True,
    ) -> NDArray[np.float64]:
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
        data: dict[str, object] = {
            "frame_number": self.frame_number,
            "tracker_type": self.tracker_type,
            "point_names": list(self.points.names),
            "xyz": self.points.xyz.tolist(),
            "visibility": self.points.visibility.tolist(),
        }
        return json.dumps(data, indent=4)

    def to_json_bytes(self) -> bytes:
        return self.to_json_string().encode("utf-8")



class BaseDetectorConfig(BaseModel, ABC):
    tracker_type: TrackerType = Field(
        description="Discriminator field — each subclass sets this to match the value of its TrackerType member.",
    )
    confidence_threshold: float = Field(
        default=0.5,
        description="Default confidence threshold for filtering tracked points (0.0-1.0)",
    )


@dataclass
class BaseImageAnnotatorConfig( ABC):
    show_overlay: bool = False


@dataclass
class BaseImageAnnotator(ABC):
    config: BaseImageAnnotatorConfig
    observations: list[BaseObservation] = field(default_factory=list)

    @classmethod
    @abstractmethod
    def create(cls, config: BaseImageAnnotatorConfig) -> "BaseImageAnnotator":
        pass

    @abstractmethod
    def annotate_image(
        self,
        image: NDArray[np.uint8],
        observation: BaseObservation,
    ) -> NDArray[np.uint8]:
        pass

    @staticmethod
    def draw_doubled_text(
        image: NDArray[np.uint8],
        text: str,
        x: int,
        y: int,
        font_scale: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness * 3)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


class BaseTrackerConfig(BaseModel, ABC):
    detector_config: BaseDetectorConfig
    annotator_config: BaseImageAnnotatorConfig | None = None


@dataclass
class BaseDetector(ABC):
    config: BaseDetectorConfig

    @classmethod
    def create(cls, config: BaseDetectorConfig) -> "BaseDetector":
        raise NotImplementedError("Must implement a method to create a detector from a config.")

    @abstractmethod
    def detect(
        self,
        frame_number: int,
        image: NDArray[np.uint8],
    ) -> BaseObservation:
        pass


@dataclass
class BaseRecorder(ABC):
    observations: list[BaseObservation] = field(default_factory=list)

    def add_observation(self, observation: BaseObservation) -> None:
        self.observations.append(observation)

    @property
    def to_array(self) -> NDArray[np.float64]:
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
        with open(output_path, "w") as json_file:
            json_file.write(self.to_json_string)

    def clear(self) -> None:
        self.observations = []


@dataclass
class BaseTracker(ABC):
    config: BaseTrackerConfig
    detector: BaseDetector
    annotator: BaseImageAnnotator
    recorder: BaseRecorder | None = None

    @classmethod
    def create(cls, config: BaseTrackerConfig) -> "BaseTracker":
        raise NotImplementedError("Must implement a method to create a tracker from a config.")

    def process_image(
        self,
        frame_number: int,
        image: NDArray[np.uint8],
        record_observation: bool = True,
    ) -> BaseObservation:
        latest_observation = self.detector.detect(image=image, frame_number=frame_number)

        if record_observation and self.recorder is not None:
            self.recorder.add_observation(observation=latest_observation)

        return latest_observation

    def annotate_image(
        self,
        image: NDArray[np.uint8],
        observation: BaseObservation,
    ) -> NDArray[np.uint8]:
        return self.annotator.annotate_image(image=image, observation=observation)

    def demo(self) -> None:
        from skellytracker.io.demo_viewers.webcam_demo_viewer import WebcamDemoViewer
        camera_viewer = WebcamDemoViewer(
            tracker=self,
            window_title=self.__class__.__name__,
        )
        camera_viewer.run()

    def image_demo(self, image_path: Path) -> None:
        from skellytracker.io.demo_viewers.image_demo_viewer import ImageDemoViewer

        image_viewer = ImageDemoViewer(self, self.__class__.__name__)
        image_viewer.run(image_path=image_path)
