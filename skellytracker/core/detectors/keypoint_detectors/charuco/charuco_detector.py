from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from beartype.typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from skellytracker.core.data_primitives import Keypoints
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import (
    KEYPOINT_DETECTOR_REGISTRY,
    KeypointDetector,
)
from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_board_definition import (
    CharucoBoardDefinition,
)
from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_detector_config import (
    CharucoDetectorConfig,
)
from skellytracker.core.detectors.metadata import EmptyMetadata
from skellytracker.core.sessions.cpu_session import CpuSession
from skellytracker.core.sessions.session import Session

logger = logging.getLogger(__name__)

MINIMUM_CHARUCO_CORNERS_FOR_VISIBILITY = 6


@dataclass
class CharucoDetector(KeypointDetector):
    """Detects charuco board corners and ArUco markers in an image.

    Returns a Keypoints instance with all charuco corners (named "CharucoCorner-{id}")
    followed by all possible ArUco marker corners (named "ArucoMarkerCorner-{id}-{j}",
    j=0..3). Undetected points are NaN with visibility=0.
    """

    config: CharucoDetectorConfig
    session: CpuSession
    _board: cv2.aruco.CharucoBoard = field(repr=False)
    _cv2_detector: cv2.aruco.CharucoDetector = field(repr=False)
    _charuco_names: tuple[str, ...] = field(repr=False)
    _aruco_ids: tuple[int, ...] = field(repr=False)
    _aruco_names: tuple[str, ...] = field(repr=False)
    _all_names: tuple[str, ...] = field(repr=False)

    def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], EmptyMetadata]:
        """Convert image to greyscale for ArUco/Charuco detection."""
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return grey, EmptyMetadata()

    def postprocess(self, raw: Any, metadata: EmptyMetadata) -> Keypoints:
        """Build Keypoints from raw detectBoard output.

        raw is a 4-tuple:
          (detected_charuco_corners, detected_charuco_ids,
           detected_aruco_corners, detected_aruco_ids)
        as returned by cv2.aruco.CharucoDetector.detectBoard().
        """
        (
            detected_charuco_corners,
            detected_charuco_ids,
            detected_aruco_corners,
            detected_aruco_ids,
        ) = raw

        detected_charuco_ids, detected_charuco_corners = _squeeze_charuco(
            detected_charuco_ids, detected_charuco_corners
        )
        detected_aruco_ids, detected_aruco_corners = _squeeze_aruco(
            detected_aruco_ids, detected_aruco_corners, self._aruco_ids
        )

        n_charuco = len(self._charuco_names)
        n_aruco_total = len(self._aruco_names)
        total = n_charuco + n_aruco_total

        xyz = np.full((total, 3), np.nan, dtype=np.float64)
        visibility = np.zeros(total, dtype=np.float64)

        if detected_charuco_ids is not None and detected_charuco_corners is not None:
            for corner_idx, corner_id in enumerate(detected_charuco_ids):
                xyz[corner_id, 0] = detected_charuco_corners[corner_idx, 0]
                xyz[corner_id, 1] = detected_charuco_corners[corner_idx, 1]
                xyz[corner_id, 2] = 0.0
                visibility[corner_id] = 1.0

        if detected_aruco_ids is not None and detected_aruco_corners is not None:
            for marker_idx, marker_id in enumerate(detected_aruco_ids):
                if marker_id not in self._aruco_ids:
                    continue
                board_marker_idx = self._aruco_ids.index(marker_id)
                for j in range(4):
                    flat_idx = n_charuco + board_marker_idx * 4 + j
                    xyz[flat_idx, 0] = detected_aruco_corners[marker_idx, j, 0]
                    xyz[flat_idx, 1] = detected_aruco_corners[marker_idx, j, 1]
                    xyz[flat_idx, 2] = 0.0
                    visibility[flat_idx] = 1.0

        return Keypoints(names=self._all_names, xyz=xyz, visibility=visibility)

    def detect(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext | None = None,
    ) -> Keypoints:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        (
            detected_charuco_corners,
            detected_charuco_ids,
            detected_aruco_corners,
            detected_aruco_ids,
        ) = self._cv2_detector.detectBoard(grey)

        detected_charuco_ids, detected_charuco_corners = _squeeze_charuco(
            detected_charuco_ids, detected_charuco_corners
        )
        detected_aruco_ids, detected_aruco_corners = _squeeze_aruco(
            detected_aruco_ids, detected_aruco_corners, self._aruco_ids
        )

        n_charuco = len(self._charuco_names)
        n_aruco_total = len(self._aruco_names)
        total = n_charuco + n_aruco_total

        xyz = np.full((total, 3), np.nan, dtype=np.float64)
        visibility = np.zeros(total, dtype=np.float64)

        # Fill charuco corners
        if detected_charuco_ids is not None and detected_charuco_corners is not None:
            for corner_idx, corner_id in enumerate(detected_charuco_ids):
                xyz[corner_id, 0] = detected_charuco_corners[corner_idx, 0]
                xyz[corner_id, 1] = detected_charuco_corners[corner_idx, 1]
                xyz[corner_id, 2] = 0.0
                visibility[corner_id] = 1.0

        # Fill ArUco marker corners
        if detected_aruco_ids is not None and detected_aruco_corners is not None:
            for marker_idx, marker_id in enumerate(detected_aruco_ids):
                if marker_id not in self._aruco_ids:
                    continue
                board_marker_idx = self._aruco_ids.index(marker_id)
                for j in range(4):
                    flat_idx = n_charuco + board_marker_idx * 4 + j
                    xyz[flat_idx, 0] = detected_aruco_corners[marker_idx, j, 0]
                    xyz[flat_idx, 1] = detected_aruco_corners[marker_idx, j, 1]
                    xyz[flat_idx, 2] = 0.0
                    visibility[flat_idx] = 1.0

        return Keypoints(names=self._all_names, xyz=xyz, visibility=visibility)

    @classmethod
    def create(cls, config: CharucoDetectorConfig, session: Session) -> CharucoDetector:
        if not isinstance(config, CharucoDetectorConfig):
            raise TypeError(f"Expected CharucoDetectorConfig, got {type(config)}")
        if not isinstance(session, CpuSession):
            raise TypeError(f"Expected CpuSession, got {type(session)}")

        board_def: CharucoBoardDefinition = config.board
        board = cv2.aruco.CharucoBoard(
            size=(board_def.squares_x, board_def.squares_y),
            squareLength=board_def.square_length_mm,
            markerLength=board_def.aruco_marker_length_mm,
            dictionary=board_def.aruco_dictionary,
        )
        cv2_detector = cv2.aruco.CharucoDetector(board)

        charuco_names = tuple(f"CharucoCorner-{i}" for i in range(board_def.n_corners))
        aruco_ids = tuple(int(i) for i in board.getIds())
        aruco_names = tuple(
            f"ArucoMarkerCorner-{marker_id}-{j}"
            for marker_id in aruco_ids
            for j in range(4)
        )

        return cls(
            config=config,
            session=session,
            _board=board,
            _cv2_detector=cv2_detector,
            _charuco_names=charuco_names,
            _aruco_ids=aruco_ids,
            _aruco_names=aruco_names,
            _all_names=charuco_names + aruco_names,
        )

    @classmethod
    def connections(cls) -> tuple[tuple[str, str], ...]:
        return ()

    def close(self) -> None:
        pass


def _squeeze_charuco(
    ids: NDArray | None,
    corners: NDArray | None,
) -> tuple[NDArray | None, NDArray | None]:
    if ids is None or corners is None:
        return None, None
    if ids.shape == (1, 1):
        ids = ids[0]
        corners = corners[0]
    else:
        ids = np.squeeze(ids)
        corners = np.squeeze(corners)
    if corners.ndim == 1:
        corners = corners[np.newaxis, :]
    return ids, corners


def _squeeze_aruco(
    ids: NDArray | None,
    corners: Sequence[NDArray] | None,
    valid_ids: tuple[int, ...],
) -> tuple[NDArray | None, NDArray | None]:
    if ids is None or corners is None:
        return None, None
    if ids.shape == (1, 1):
        ids = ids[0]
    else:
        ids = np.squeeze(ids)
    corners_squeezed = [np.squeeze(c) for c in corners]

    valid_pairs = [
        (marker_id, c)
        for marker_id, c in zip(ids.ravel(), corners_squeezed, strict=False)
        if int(marker_id) in valid_ids
    ]
    if not valid_pairs:
        return None, None

    filtered_ids = np.array([p[0] for p in valid_pairs], dtype=np.int32)
    filtered_corners = np.stack([p[1] for p in valid_pairs])  # (M, 4, 2)
    return filtered_ids, filtered_corners


KEYPOINT_DETECTOR_REGISTRY["charuco"] = CharucoDetector
