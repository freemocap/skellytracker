import unittest
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_board_selection import CharucoBoardSelector, StandardCharucoBoard


class CharucoSearchScheduleTests(unittest.TestCase):
    def test_misses_skip_increasing_frames_up_to_cap(self) -> None:
        selector = CharucoBoardSelector(maximum_skipped_frames=3)
        searched: list[int] = []
        blank = np.full((240, 320), 255, dtype=np.uint8)
        for frame in range(18):
            def images() -> Iterator[NDArray[np.uint8]]:
                searched.append(frame)
                yield blank
            self.assertIsNone(selector.search_frame(frame_number=frame, images=images()))
        self.assertEqual(searched, [0, 2, 5, 9, 13, 17])

    def test_first_board_in_group_locks_without_consuming_later_images(self) -> None:
        selector = CharucoBoardSelector()
        board = StandardCharucoBoard.TEST_7X5.board_definition()
        blank = np.full((240, 320), 255, dtype=np.uint8)
        def images() -> Iterator[NDArray[np.uint8]]:
            yield blank
            yield board.cv2_board.generateImage((840, 600), marginSize=30)
            raise AssertionError("No more camera images may be requested after selection")
        self.assertEqual(selector.search_frame(frame_number=0, images=images()), board)
        self.assertEqual(selector.search_frame(frame_number=1, images=()), board)

    def test_invalid_frame_order_is_an_error(self) -> None:
        selector = CharucoBoardSelector()
        blank = np.full((240, 320), 255, dtype=np.uint8)
        selector.search_frame(frame_number=0, images=[blank])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            selector.search_frame(frame_number=0, images=[blank])
