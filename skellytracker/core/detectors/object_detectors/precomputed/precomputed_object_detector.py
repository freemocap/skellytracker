from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.config.detector_configs import ObjectDetectorConfig
from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import ObjectDetector
from skellytracker.core.sessions.cpu_session import CpuSession
from skellytracker.core.sessions.session import Session


class PrecomputedObjectDetectorConfig(ObjectDetectorConfig):
    detector_type: Literal["precomputed"] = "precomputed"
    session_backend: Literal["cpu"] = "cpu"


@dataclass
class PrecomputedObjectDetector(ObjectDetector):
    """ObjectDetector that returns pre-computed bounding boxes by frame number.

    Useful for testing or benchmarking keypoint detectors independently of
    the object detection step — supply boxes from a previous run or a
    ground-truth source and skip redundant inference.

    Falls back to a full-image bounding box when a frame number is not found
    in the supplied dict.
    """

    config: PrecomputedObjectDetectorConfig = field(
        default_factory=PrecomputedObjectDetectorConfig
    )
    session: Session = field(default_factory=CpuSession)
    bboxes_by_frame: dict[int, list[BoundingBox]] = field(default_factory=dict)

    def detect(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext | None = None,
    ) -> list[BoundingBox]:
        if context is not None and context.frame_number in self.bboxes_by_frame:
            return self.bboxes_by_frame[context.frame_number]
        h, w = image.shape[:2]
        return [BoundingBox.full_image(h, w)]

    @classmethod
    def create(
        cls,
        config: ObjectDetectorConfig,
        session: Session,
    ) -> PrecomputedObjectDetector:
        return cls(config=config, session=session)

    @classmethod
    def from_list(
        cls,
        bboxes: list[list[BoundingBox]],
    ) -> PrecomputedObjectDetector:
        """Create from a list where the list index is the frame number."""
        return cls(bboxes_by_frame={i: boxes for i, boxes in enumerate(bboxes)})
