import unittest
from collections.abc import Iterator
from unittest.mock import Mock

import cv2
import numpy as np
from numpy.typing import NDArray

from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_board_selection import (
    CharucoBoardSelectionError, CharucoBoardSelector, StandardCharucoBoard,
)


def board_image(preset: StandardCharucoBoard) -> NDArray[np.uint8]:
    board = preset.board_definition()
    return board.cv2_board.generateImage((board.squares_x * 120, board.squares_y * 120), marginSize=30)


class CharucoBoardSelectionTests(unittest.TestCase):
    def test_each_standard_board_under_perspective(self) -> None:
        for preset in StandardCharucoBoard:
            with self.subTest(preset=preset):
                image = board_image(preset)
                height, width = image.shape
                source = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
                images: list[NDArray[np.uint8]] = []
                for offset in (5, 15, 25):
                    target = source + np.float32([[offset, offset], [-offset, 0], [0, -offset], [offset, 0]])
                    images.append(cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, target), (width, height), borderValue=255))
                selected = CharucoBoardSelector.from_images(images=images)
                self.assertEqual(selected, preset.board_definition())

    def test_boardless_frames_do_not_change_selection(self) -> None:
        blank = np.full((480, 640), 255, dtype=np.uint8)
        image = board_image(StandardCharucoBoard.LETTER_5X3)
        selected = CharucoBoardSelector.from_images(images=[blank, image, blank, image, image, blank])
        self.assertEqual(selected.squares_x, 5)

    def test_first_match_stops_consuming_images(self) -> None:
        for preset in StandardCharucoBoard:
            def images() -> Iterator[NDArray[np.uint8]]:
                yield board_image(preset)
                raise AssertionError("Images must not be requested after the first match")
            self.assertEqual(CharucoBoardSelector.from_images(images=images()), preset.board_definition())

    def test_locked_selection_does_not_run_either_detector(self) -> None:
        selector = CharucoBoardSelector()
        selected = selector.observe_image(image=board_image(StandardCharucoBoard.TEST_7X5))
        selector._marker_detector = Mock()
        self.assertEqual(selector.observe_image(image=board_image(StandardCharucoBoard.LETTER_5X3)), selected)
        selector._marker_detector.detectMarkers.assert_not_called()

    def test_first_candidate_match_does_not_run_second_candidate(self) -> None:
        selector = CharucoBoardSelector()
        second_detector = Mock()
        selector._board_detectors[StandardCharucoBoard.TEST_7X5] = second_detector
        selected = selector.observe_image(image=board_image(StandardCharucoBoard.LETTER_5X3))
        self.assertEqual(selected, StandardCharucoBoard.LETTER_5X3.board_definition())
        second_detector.detectBoard.assert_not_called()

    def test_no_board_produces_explicit_error(self) -> None:
        images = [np.full((480, 640), 255, dtype=np.uint8)]
        with self.assertRaisesRegex(CharucoBoardSelectionError, "could not detect"):
            CharucoBoardSelector.from_images(images=images)
