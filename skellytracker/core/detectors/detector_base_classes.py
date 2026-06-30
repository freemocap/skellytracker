from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np

from skellytracker.core.config.detector_configs import (
    KeypointDetectorConfig,
    ObjectDetectorConfig,
)
from skellytracker.core.data_primitives import BoundingBox, Keypoints
from skellytracker.core.session import Session


@dataclass
class ObjectDetector(ABC):
    """Detects objects in an image and returns bounding boxes.

    Stateless between calls — all temporal state lives in TrackerState.
    Does not own GPU resources; receives a Session at construction time.
    """

    config: ObjectDetectorConfig
    session: Session

    @abstractmethod
    def detect(self, image: NDArray[np.uint8]) -> list[BoundingBox]:
        """Run detection on an image and return zero or more bounding boxes."""
        ...

    @classmethod
    @abstractmethod
    def create(cls, config: ObjectDetectorConfig, session: Session) -> ObjectDetector:
        ...


@dataclass
class KeypointDetector(ABC):
    """Estimates keypoints on a (cropped) image.

    Stateless between calls — all temporal state lives in TrackerState.
    Does not own GPU resources; receives a Session at construction time.
    Point names and ordering are defined by a YAML-defined schema.
    """

    config: KeypointDetectorConfig
    session: Session

    @abstractmethod
    def detect(
        self,
        image: NDArray[np.uint8],
        bbox: BoundingBox | None = None,
    ) -> Keypoints:
        """Run keypoint estimation and return named points with visibility scores."""
        ...

    @classmethod
    @abstractmethod
    def create(cls, config: KeypointDetectorConfig, session: Session) -> KeypointDetector:
        ...


OBJECT_DETECTOR_REGISTRY: dict[str, type[ObjectDetector]] = {}
KEYPOINT_DETECTOR_REGISTRY: dict[str, type[KeypointDetector]] = {}


def build_object_detector(
    config: ObjectDetectorConfig,
    sessions: dict[str, Session],
) -> ObjectDetector:
    """Instantiate an ObjectDetector from config using the registry."""
    cls = OBJECT_DETECTOR_REGISTRY.get(config.detector_type)
    if cls is None:
        raise KeyError(
            f"No ObjectDetector registered for type {config.detector_type!r}. "
            f"Registered types: {list(OBJECT_DETECTOR_REGISTRY)}"
        )
    session = sessions.get(config.session_backend)
    return cls.create(config, session)


def build_keypoint_detector(
    config: KeypointDetectorConfig,
    sessions: dict[str, Session],
) -> KeypointDetector:
    """Instantiate a KeypointDetector from config using the registry."""
    cls = KEYPOINT_DETECTOR_REGISTRY.get(config.detector_type)
    if cls is None:
        raise KeyError(
            f"No KeypointDetector registered for type {config.detector_type!r}. "
            f"Registered types: {list(KEYPOINT_DETECTOR_REGISTRY)}"
        )
    session = sessions.get(config.session_backend)
    return cls.create(config, session)
