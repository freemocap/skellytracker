from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from skellytracker.core.data_primitives import Keypoints
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import (
    KEYPOINT_DETECTOR_REGISTRY,
    KeypointDetector,
)
from skellytracker.core.detectors.keypoint_detectors.aruco.aruco_detector_config import (
    ArucoDetectorConfig,
)
from skellytracker.core.sessions.cpu_session import CpuSession
from skellytracker.core.sessions.session import Session

logger = logging.getLogger(__name__)


@dataclass
class ArucoDetector(KeypointDetector):
    """Detects ArUco markers in an image and returns their 4 corner keypoints.

    Returns a Keypoints instance with 4 corners per configured marker ID
    (named "ArucoMarker-{id}-corner-{j}", j=0..3). Undetected markers have
    NaN coordinates and visibility=0.
    """

    config: ArucoDetectorConfig
    session: CpuSession
    _cv2_detector: cv2.aruco.ArucoDetector = field(repr=False)
    _aruco_ids: tuple[int, ...] = field(repr=False)
    _all_names: tuple[str, ...] = field(repr=False)

    def detect(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext | None = None,
    ) -> Keypoints:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        detected_corners, detected_ids, _ = self._cv2_detector.detectMarkers(grey)

        n_total = len(self._aruco_ids) * 4
        xyz = np.full((n_total, 3), np.nan, dtype=np.float64)
        visibility = np.zeros(n_total, dtype=np.float64)

        if detected_ids is not None and len(detected_corners) > 0:
            detected_ids_flat = detected_ids.ravel()
            for det_idx, marker_id in enumerate(detected_ids_flat):
                marker_id = int(marker_id)
                if marker_id not in self._aruco_ids:
                    continue
                config_idx = self._aruco_ids.index(marker_id)
                corners = np.squeeze(detected_corners[det_idx])  # (4, 2)
                if corners.ndim == 1:
                    corners = corners[np.newaxis, :]
                for j in range(4):
                    flat_idx = config_idx * 4 + j
                    xyz[flat_idx, 0] = corners[j, 0]
                    xyz[flat_idx, 1] = corners[j, 1]
                    xyz[flat_idx, 2] = 0.0
                    visibility[flat_idx] = 1.0

        return Keypoints(names=self._all_names, xyz=xyz, visibility=visibility)

    @classmethod
    def create(cls, config: ArucoDetectorConfig, session: Session) -> ArucoDetector:
        if not isinstance(config, ArucoDetectorConfig):
            raise TypeError(f"Expected ArucoDetectorConfig, got {type(config)}")
        if not isinstance(session, CpuSession):
            raise TypeError(f"Expected CpuSession, got {type(session)}")

        dictionary = cv2.aruco.getPredefinedDictionary(config.aruco_dictionary_enum)
        params = cv2.aruco.DetectorParameters()
        cv2_detector = cv2.aruco.ArucoDetector(dictionary, params)

        aruco_ids = config.aruco_ids
        all_names = tuple(
            f"ArucoMarker-{marker_id}-corner-{j}"
            for marker_id in aruco_ids
            for j in range(4)
        )

        return cls(
            config=config,
            session=session,
            _cv2_detector=cv2_detector,
            _aruco_ids=aruco_ids,
            _all_names=all_names,
        )

    @classmethod
    def connections(cls) -> tuple[tuple[str, str], ...]:
        return ()

    def close(self) -> None:
        pass


KEYPOINT_DETECTOR_REGISTRY["aruco"] = ArucoDetector
