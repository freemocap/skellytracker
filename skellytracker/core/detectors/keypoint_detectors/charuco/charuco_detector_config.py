from __future__ import annotations

from typing import Literal

from pydantic import Field

from skellytracker.core.config.detector_configs import KeypointDetectorConfig
from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_board_definition import (
    CharucoBoardDefinition,
)


class CharucoDetectorConfig(KeypointDetectorConfig):
    detector_type: Literal["charuco"] = "charuco"
    session_backend: Literal["cpu"] = "cpu"
    board: CharucoBoardDefinition = Field(
        default_factory=CharucoBoardDefinition.create_letter_size_5x3
    )
