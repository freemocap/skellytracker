from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray

from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_board_definition import CharucoBoardDefinition


class StandardCharucoBoard(StrEnum):
    LETTER_5X3 = "5x3"
    TEST_7X5 = "7x5"

    def board_definition(self) -> CharucoBoardDefinition:
        match self:
            case StandardCharucoBoard.LETTER_5X3:
                return CharucoBoardDefinition.create_letter_size_5x3()
            case StandardCharucoBoard.TEST_7X5:
                return CharucoBoardDefinition.create_test_data_7x5()


class CharucoBoardSelectionError(ValueError):
    """Images do not establish exactly one supported calibration board."""


@dataclass
class CharucoBoardSelector:
    """Lock the first supported board, testing 5x3 before 7x5 in each image.

    A valid match has six non-collinear interpolated corners. Blank images
    leave selection unresolved. Once selected, no further detection is done.
    Square lengths come from the standard definitions, never from image scale.
    """

    maximum_skipped_frames: int = 30
    _miss_count: int = field(default=0, init=False)
    _next_search_frame: int = field(default=0, init=False)
    _last_search_input_frame: int = field(default=-1, init=False)
    _marker_detector: cv2.aruco.ArucoDetector = field(init=False, repr=False)
    _board_detectors: dict[StandardCharucoBoard, cv2.aruco.CharucoDetector] = field(init=False, repr=False)
    _selected_board: CharucoBoardDefinition | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.maximum_skipped_frames < 1:
            raise ValueError("Maximum skipped frames must be positive")
        self._marker_detector = cv2.aruco.ArucoDetector(
            StandardCharucoBoard.LETTER_5X3.board_definition().aruco_dictionary,
        )
        self._board_detectors = {
            preset: cv2.aruco.CharucoDetector(preset.board_definition().cv2_board)
            for preset in StandardCharucoBoard
        }

    def observe_image(self, *, image: NDArray[np.uint8]) -> CharucoBoardDefinition | None:
        if self._selected_board is not None:
            return self._selected_board
        if image.dtype != np.uint8 or image.ndim not in (2, 3) or image.size == 0:
            raise ValueError("Board selection requires a nonempty uint8 grayscale or BGR image")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        marker_corners, marker_ids, _ = self._marker_detector.detectMarkers(gray)
        if marker_ids is None:
            return None
        for preset, detector in self._board_detectors.items():
            corners, corner_ids, _, _ = detector.detectBoard(
                gray, markerCorners=marker_corners, markerIds=marker_ids,
            )
            if corner_ids is None or len(corner_ids) < 6:
                continue
            board_points = preset.board_definition().cv2_board.getChessboardCorners()[corner_ids.reshape(-1), :2]
            if np.linalg.matrix_rank(board_points - board_points.mean(axis=0)) < 2:
                continue
            self._selected_board = preset.board_definition()
            return self._selected_board
        return None

    def search_frame(
        self, *, frame_number: int, images: Iterable[NDArray[np.uint8]],
    ) -> CharucoBoardDefinition | None:
        """Search a synchronized frame with capped linear backoff after misses.

        Search frame 0, then skip 1, 2, 3, ... frames after successive misses.
        A miss means none of the supplied camera images establishes a board.
        Skipped frames and locked selections do not consume the image iterable.
        Call in strictly increasing multiframe order; body processing is independent.
        """
        if frame_number < 0 or frame_number <= self._last_search_input_frame:
            raise ValueError("Board search requires strictly increasing nonnegative frame numbers")
        self._last_search_input_frame = frame_number
        if self._selected_board is not None:
            return self._selected_board
        if frame_number < self._next_search_frame:
            return None
        received_image = False
        for image in images:
            received_image = True
            board = self.observe_image(image=image)
            if board is not None:
                return board
        if not received_image:
            raise ValueError("Board search requires at least one camera image")
        self._miss_count += 1
        self._next_search_frame = frame_number + min(self._miss_count, self.maximum_skipped_frames) + 1
        return None

    @classmethod
    def from_images(cls, *, images: Iterable[NDArray[np.uint8]]) -> CharucoBoardDefinition:
        selector = cls()
        for image in images:
            selected_board = selector.observe_image(image=image)
            if selected_board is not None:
                return selected_board
        raise CharucoBoardSelectionError(
            "AUTO could not detect a supported 5x3 or 7x5 calibration board in the recording."
        )
