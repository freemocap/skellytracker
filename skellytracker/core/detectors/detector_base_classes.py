from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray
import numpy as np

from skellytracker.core.config.detector_configs import (
    KeypointDetectorConfig,
    ObjectDetectorConfig,
)
from skellytracker.core.data_primitives import Keypoints
from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.sessions.session import Session


@dataclass
class ObjectDetector(ABC):
    """Detects objects in an image and returns bounding boxes.

    Stateless between calls — all temporal state lives in TrackerState.
    Receives a Session at construction time for shared device context.
    """

    config: ObjectDetectorConfig
    session: Session

    @abstractmethod
    def detect(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext | None = None,
    ) -> list[BoundingBox]:
        """Run detection on an image and return zero or more bounding boxes."""
        ...

    @abstractmethod
    def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], Any]:
        """Prepare image for inference. Returns (tensor, metadata).

        The tensor has shape (3, H, W) and dtype float32. Metadata carries any
        information needed by postprocess to decode raw model outputs back to
        image-space coordinates (e.g. letterbox ratio, original size).
        """
        ...

    @abstractmethod
    def postprocess(self, raw: Any, metadata: Any) -> list[BoundingBox]:
        """Decode raw inference output back to image-space bounding boxes.

        raw is the per-image split output from session.run or run_batched —
        a list of arrays where each array corresponds to one ORT output.
        metadata is whatever preprocess returned as its second element.
        """
        ...

    @classmethod
    @abstractmethod
    def create(cls, config: ObjectDetectorConfig, session: Session) -> ObjectDetector:
        ...

    def close(self) -> None:  # noqa: B027
        """Release any resources owned by this detector. Override if needed."""

    def reset_temporal_state(self) -> None:  # noqa: B027
        """Reset any internal temporal state (e.g. tracking history).

        Called by process_folder between videos so each video is processed
        from a clean state. Default is a no-op; override for stateful backends.
        """

    @classmethod
    def connections(cls) -> tuple[tuple[str, str], ...]:
        """Return skeleton connection pairs as (name_a, name_b) tuples for annotation."""
        return ()


@dataclass
class KeypointDetector(ABC):
    """Estimates keypoints on a (cropped) image.

    Stateless between calls — all temporal state lives in TrackerState.
    Receives a Session at construction time for shared device context.
    Point names and ordering are defined by a YAML-defined schema.
    """

    config: KeypointDetectorConfig
    session: Session

    @abstractmethod
    def detect(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext | None = None,
    ) -> Keypoints:
        """Run keypoint estimation on image and return named points with visibility scores.

        image must already be cropped to the region of interest. Coordinates are
        returned in image-local space; the caller is responsible for translating
        them back to full-frame space.
        """
        ...

    @abstractmethod
    def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], Any]:
        """Prepare image for inference. Returns (tensor, metadata).

        The tensor has shape (3, H, W) and dtype float32. Metadata carries any
        information needed by postprocess to reconstruct image-space coordinates
        (e.g. center/scale for RTMPose).

        For non-ONNX detectors the tensor may carry a different dtype (e.g.
        uint8 for MediaPipe), but the second element is always the metadata
        object needed by postprocess.
        """
        ...

    @abstractmethod
    def postprocess(self, raw: Any, metadata: Any) -> Keypoints:
        """Decode raw inference output back to image-space keypoints.

        raw is the per-image split output from session.run or run_batched.
        metadata is whatever preprocess returned as its second element.
        """
        ...

    @classmethod
    @abstractmethod
    def create(cls, config: KeypointDetectorConfig, session: Session) -> KeypointDetector:
        ...

    def close(self) -> None:  # noqa: B027
        """Release any resources owned by this detector. Override if needed."""

    def reset_temporal_state(self) -> None:  # noqa: B027
        """Reset any internal temporal state (e.g. tracking history).

        Called by process_folder between videos so each video is processed
        from a clean state. Default is a no-op; override for stateful backends.
        """

    @classmethod
    def connections(cls) -> tuple[tuple[str, str], ...]:
        """Return skeleton connection pairs as (name_a, name_b) tuples for annotation."""
        return ()


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
