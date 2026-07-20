from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.annotation.annotator import Annotator
from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_annotator import (
    CharucoAnnotator,
    CharucoAnnotatorConfig,
)
from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_board_definition import (
    CharucoBoardDefinition,
)
from skellytracker.core.data_primitives.observation import Observation


@dataclass
class CharucoObservationAnnotator(Annotator):
    """Bridges the Annotator interface to CharucoAnnotator.

    Pulls the charuco Keypoints out of the named stage in the Observation and
    delegates drawing to CharucoAnnotator.
    """

    _inner: CharucoAnnotator
    stage_name: str = "charuco"

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
    def create(cls, config: object) -> CharucoObservationAnnotator:
        if not isinstance(config, _CharucoObservationAnnotatorConfig):
            raise TypeError(
                f"Expected _CharucoObservationAnnotatorConfig, got {type(config).__name__}"
            )
        inner = CharucoAnnotator(config=config.annotator_config, board_def=config.board_def)
        return cls(_inner=inner, stage_name=config.stage_name)


@dataclass
class _CharucoObservationAnnotatorConfig:
    board_def: CharucoBoardDefinition
    annotator_config: CharucoAnnotatorConfig = field(default_factory=CharucoAnnotatorConfig)
    stage_name: str = "charuco"
