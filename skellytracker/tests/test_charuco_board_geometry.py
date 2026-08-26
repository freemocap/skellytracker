"""The board's geometry and names, single-sourced on the cv2 board it describes.

`CharucoBoardDefinition` is the one place that knows what a board's points are called and
where they sit. The detector reads the names from here, the calibration solver reads the
corner positions from here, and skellyforge's board skeleton is built from the normalized
positions and connections here. Two of those used to have their own copies.
"""

from __future__ import annotations

import numpy as np
import pytest

from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_board_definition import (
    CORNERS_PER_ARUCO_MARKER,
    CharucoBoardDefinition,
)

_BOARDS = {
    "5x3": CharucoBoardDefinition.create_letter_size_5x3(),
    "7x5": CharucoBoardDefinition.create_test_data_7x5(),
    # Not a shipped default: the geometry is parametric, so a board a user prints must
    # work with no code change.
    "9x7": CharucoBoardDefinition(squares_x=9, squares_y=7, square_length_mm=33.0),
}


def _legacy_corner_positions(board: CharucoBoardDefinition) -> np.ndarray:
    """The hand-rolled grid `corner_positions_board_frame` used to generate itself."""
    columns = board.squares_x - 1
    rows = board.squares_y - 1
    positions = np.zeros((columns * rows, 3), dtype=np.float64)
    positions[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    return positions * board.square_length_mm


@pytest.mark.parametrize("board", _BOARDS.values(), ids=_BOARDS.keys())
def test_corner_positions_still_match_the_values_they_replaced(
    board: CharucoBoardDefinition,
) -> None:
    """De-duplication, not a change: the calibration's object frame must not move.

    OpenCV measures from the board's outer edge, so its first interior corner sits one
    square in — a constant offset across every corner. Subtracting it reproduces the
    previous values exactly, which is what makes deriving them from cv2 safe.
    """
    np.testing.assert_array_equal(
        board.corner_positions_board_frame, _legacy_corner_positions(board)
    )


@pytest.mark.parametrize("board", _BOARDS.values(), ids=_BOARDS.keys())
def test_every_named_point_has_a_position_and_vice_versa(
    board: CharucoBoardDefinition,
) -> None:
    positions = board.normalized_point_positions
    assert set(positions) == set(board.all_point_names)
    assert len(board.charuco_corner_names) == board.n_corners
    assert (
        len(board.aruco_corner_names)
        == len(board.aruco_marker_ids) * CORNERS_PER_ARUCO_MARKER
    )


@pytest.mark.parametrize("board", _BOARDS.values(), ids=_BOARDS.keys())
def test_positions_are_normalized_to_the_square_length(
    board: CharucoBoardDefinition,
) -> None:
    """`1.0` is one square — the board's reference unit, like body height for the human.

    So a fit that recovers a scale of 54.0 is saying "the squares measure 54mm", which is
    directly comparable to the value entered at calibration.
    """
    positions = np.array(list(board.normalized_point_positions.values()))
    # A board is flat, and spans (squares - 1) squares between its outermost corners.
    np.testing.assert_allclose(positions[:, 2], 0.0, atol=1e-9)
    span = np.ptp(positions, axis=0)
    assert span[0] == pytest.approx(board.squares_x - 0.2, abs=0.05)
    assert span[1] == pytest.approx(board.squares_y - 0.2, abs=0.05)

    # Neighbouring interior corners are exactly one square apart.
    corners = np.array(
        [board.normalized_point_positions[name] for name in board.charuco_corner_names]
    )
    assert float(np.linalg.norm(corners[1] - corners[0])) == pytest.approx(1.0)


@pytest.mark.parametrize("board", _BOARDS.values(), ids=_BOARDS.keys())
def test_connections_name_only_real_points(board: CharucoBoardDefinition) -> None:
    """The structure a consumer draws, so it never has to rebuild the grid itself."""
    known = set(board.all_point_names)
    for first, second in board.charuco_grid_connections:
        assert first in known and second in known
        assert first != second
    for first, second in board.aruco_marker_connections:
        assert first in known and second in known
        assert first != second


@pytest.mark.parametrize("board", _BOARDS.values(), ids=_BOARDS.keys())
def test_every_marker_gets_a_closed_quad(board: CharucoBoardDefinition) -> None:
    assert (
        len(board.aruco_marker_connections)
        == len(board.aruco_marker_ids) * CORNERS_PER_ARUCO_MARKER
    )


def test_the_grid_joins_each_corner_to_its_neighbours() -> None:
    """A 5x3 board has a 4x2 lattice of interior corners: 4 horizontal + 6 vertical edges."""
    board = _BOARDS["5x3"]
    columns, rows = board.squares_x - 1, board.squares_y - 1
    expected = rows * (columns - 1) + columns * (rows - 1)
    assert len(board.charuco_grid_connections) == expected
