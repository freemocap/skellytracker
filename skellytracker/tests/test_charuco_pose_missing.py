"""Missing board evidence is a normal per-image result."""
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from skellytracker.core.data_primitives import Keypoints
from skellytracker.core.detectors.keypoint_detectors.charuco import CharucoBoardDefinition
from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_board_pose import compute_board_pose


class MissingBoardPoseTests(TestCase):
    def test_missing_or_partial_board_returns_none_without_warning(self):
        board = CharucoBoardDefinition.create_letter_size_5x3()
        for count in (0, 5):
            with self.subTest(count=count):
                xyz = np.full((board.n_corners, 3), np.nan)
                xyz[:count] = 1.
                visibility = np.zeros(board.n_corners)
                visibility[:count] = 1.
                points = Keypoints(names=board.charuco_corner_names, xyz=xyz, visibility=visibility)
                with self.assertNoLogs(level='WARNING'):
                    self.assertIsNone(compute_board_pose(points, board, np.eye(3), np.zeros(5)))

    def test_unsolved_pose_returns_none_without_warning(self):
        board = CharucoBoardDefinition.create_letter_size_5x3()
        points = Keypoints(names=board.charuco_corner_names,
            xyz=np.ones((board.n_corners, 3)), visibility=np.ones(board.n_corners))
        with patch('cv2.solvePnP', return_value=(False, None, None)), self.assertNoLogs(level='WARNING'):
            self.assertIsNone(compute_board_pose(points, board, np.eye(3), np.zeros(5)))
