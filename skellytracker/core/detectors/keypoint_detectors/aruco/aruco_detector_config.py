from __future__ import annotations

from typing import Literal

import cv2
from pydantic import Field

from skellytracker.core.config.detector_configs import KeypointDetectorConfig


class ArucoDetectorConfig(KeypointDetectorConfig):
    detector_type: Literal["aruco"] = "aruco"
    session_backend: Literal["cpu"] = "cpu"
    aruco_ids: tuple[int, ...] = (0, 1, 2, 3)
    aruco_dictionary_enum: int = Field(default=cv2.aruco.DICT_4X4_50)
