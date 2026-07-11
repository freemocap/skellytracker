from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.annotation.annotator import Annotator
from skellytracker.core.detectors.keypoint_detectors.aruco.aruco_annotator import (
    ArucoAnnotator,
    ArucoAnnotatorConfig,
)
from skellytracker.core.observation import Observation


@dataclass
class ArucoObservationAnnotator(Annotator):
    """Bridges the Annotator interface to ArucoAnnotator.

    Pulls the aruco Keypoints out of the named stage in the Observation and
    delegates drawing to ArucoAnnotator.
    """

    _inner: ArucoAnnotator
    stage_name: str = "aruco"

    def annotate(
        self,
        image: NDArray[np.uint8],
        observation: Observation,
    ) -> NDArray[np.uint8]:
        stage = observation.stages.get(self.stage_name)
        if stage is None or stage.keypoints is None:
            return image.copy()
        return self._inner.annotate(image, stage.keypoints)

    @classmethod
    def create(cls, config: object) -> ArucoObservationAnnotator:
        if not isinstance(config, _ArucoObservationAnnotatorConfig):
            raise TypeError(
                f"Expected _ArucoObservationAnnotatorConfig, got {type(config).__name__}"
            )
        inner = ArucoAnnotator(
            config=config.annotator_config,
            aruco_ids=config.aruco_ids,
        )
        return cls(_inner=inner, stage_name=config.stage_name)


@dataclass
class _ArucoObservationAnnotatorConfig:
    aruco_ids: tuple[int, ...]
    annotator_config: ArucoAnnotatorConfig = field(default_factory=ArucoAnnotatorConfig)
    stage_name: str = "aruco"
