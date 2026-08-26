import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, computed_field, model_validator

CORNERS_PER_ARUCO_MARKER: int = 4

# What this board's points and edges ARE, for whoever builds its skeleton. Tags, not
# colours: skellyforge's `definitions/color_palette.yaml` turns a tag into a colour, and a
# user recolours by editing that one mapping. Ordered most-specific-first, because a
# palette resolves the first tag it knows.
#
# They are named here, next to the geometry they describe, so the board has one place that
# says what its parts are. They are plain strings, so this file gains no dependency on the
# package that draws them.
CHARUCO_CORNER_TAGS: tuple[str, ...] = ("charuco_corner",)
CHARUCO_GRID_TAGS: tuple[str, ...] = ("charuco_grid", "charuco_corner")
ARUCO_MARKER_TAGS: tuple[str, ...] = ("aruco_marker",)


class CharucoBoardDefinition(BaseModel):
    """Known charuco board geometry — fixed, never optimized.

    Single source of truth for board parameters. Both the detector and
    calibration solver construct their cv2-specific representations from this.
    """

    model_config = ConfigDict(extra="forbid")

    squares_x: int
    squares_y: int
    square_length_mm: float
    marker_length_ratio: float = 0.8
    aruco_dictionary_enum: int = cv2.aruco.DICT_4X4_250

    @computed_field
    @property
    def aruco_marker_length_mm(self) -> float:
        return self.marker_length_ratio * self.square_length_mm

    @model_validator(mode="after")
    def validate_geometry(self) -> "CharucoBoardDefinition":
        if self.aruco_marker_length_mm >= self.square_length_mm:
            raise ValueError(
                f"aruco_marker_length_mm ({self.aruco_marker_length_mm}) must be < "
                f"square_length_mm ({self.square_length_mm})"
            )
        if self.squares_x < 2 or self.squares_y < 2:
            raise ValueError(
                f"Board must have at least 2x2 squares, "
                f"got {self.squares_x}x{self.squares_y}"
            )
        return self

    @property
    def aruco_dictionary(self) -> cv2.aruco.Dictionary:
        return cv2.aruco.getPredefinedDictionary(self.aruco_dictionary_enum)

    @property
    def cv2_board(self) -> cv2.aruco.CharucoBoard:
        """The OpenCV board this definition describes.

        Built here rather than by each caller so the detector, the calibration solver and
        the geometry accessors below cannot disagree about what the board is. OpenCV owns
        the marker ids and the corner ordering; asking it is what keeps the names attached
        to the right points.
        """
        return cv2.aruco.CharucoBoard(
            size=(self.squares_x, self.squares_y),
            squareLength=self.square_length_mm,
            markerLength=self.aruco_marker_length_mm,
            dictionary=self.aruco_dictionary,
        )

    @property
    def n_corners(self) -> int:
        """Number of internal charuco corners."""
        return (self.squares_x - 1) * (self.squares_y - 1)

    # ── Names ──────────────────────────────────────────────────────────
    # The single source of truth for what this board's points are called. The detector
    # reads these rather than rebuilding the f-strings, so a rename cannot leave the
    # detector emitting one name and the model expecting another.

    @property
    def charuco_corner_names(self) -> tuple[str, ...]:
        """The interior chessboard corners, in OpenCV's corner-id order."""
        return tuple(f"CharucoCorner-{index}" for index in range(self.n_corners))

    @property
    def aruco_marker_ids(self) -> tuple[int, ...]:
        """The dictionary ids of the markers OpenCV places on this board."""
        return tuple(int(marker_id) for marker_id in self.cv2_board.getIds())

    @property
    def aruco_corner_names(self) -> tuple[str, ...]:
        """Every marker's four corners, in OpenCV's per-marker corner order."""
        return tuple(
            f"ArucoMarkerCorner-{marker_id}-{corner}"
            for marker_id in self.aruco_marker_ids
            for corner in range(CORNERS_PER_ARUCO_MARKER)
        )

    @property
    def all_point_names(self) -> tuple[str, ...]:
        """Every point this board can produce: charuco corners, then marker corners."""
        return self.charuco_corner_names + self.aruco_corner_names

    # ── Geometry, normalized ───────────────────────────────────────────
    # Positions as multiples of the square length, so `1.0` is one square. This is the
    # board's REFERENCE UNIT, the way body height is the standard human's: it makes the
    # model size-agnostic, and it makes the fitted scale mean "the measured square length",
    # which is directly comparable to the value entered at calibration.

    @property
    def normalized_point_positions(self) -> dict[str, NDArray[np.float64]]:
        """Every named point's `(3,)` position on the board, in square-length units."""
        positions: dict[str, NDArray[np.float64]] = {}
        chessboard_corners = np.asarray(
            self.cv2_board.getChessboardCorners(), dtype=np.float64
        )
        for name, position in zip(self.charuco_corner_names, chessboard_corners):
            positions[name] = position / self.square_length_mm
        marker_object_points = self.cv2_board.getObjPoints()
        for marker_id, marker_corners in zip(
            self.aruco_marker_ids, marker_object_points
        ):
            corners = np.asarray(marker_corners, dtype=np.float64).reshape(-1, 3)
            for corner_index, corner in enumerate(corners):
                positions[f"ArucoMarkerCorner-{marker_id}-{corner_index}"] = (
                    corner / self.square_length_mm
                )
        return positions

    @property
    def charuco_grid_connections(self) -> tuple[tuple[str, str], ...]:
        """The chessboard lattice: each interior corner joined to its right and below.

        Declared here so no consumer has to rebuild the grid from board dimensions, which
        is the client-side derivation this replaces.
        """
        columns = self.squares_x - 1
        rows = self.squares_y - 1
        pairs: list[tuple[str, str]] = []
        for row in range(rows):
            for column in range(columns):
                index = row * columns + column
                if column < columns - 1:
                    pairs.append(
                        (f"CharucoCorner-{index}", f"CharucoCorner-{index + 1}")
                    )
                if row < rows - 1:
                    pairs.append(
                        (f"CharucoCorner-{index}", f"CharucoCorner-{index + columns}")
                    )
        return tuple(pairs)

    @property
    def aruco_marker_connections(self) -> tuple[tuple[str, str], ...]:
        """Each marker's four sides, as a closed quad around its corners."""
        pairs: list[tuple[str, str]] = []
        for marker_id in self.aruco_marker_ids:
            for corner in range(CORNERS_PER_ARUCO_MARKER):
                next_corner = (corner + 1) % CORNERS_PER_ARUCO_MARKER
                pairs.append(
                    (
                        f"ArucoMarkerCorner-{marker_id}-{corner}",
                        f"ArucoMarkerCorner-{marker_id}-{next_corner}",
                    )
                )
        return tuple(pairs)

    @property
    def corner_positions_board_frame(self) -> NDArray[np.float64]:
        """(n_corners, 3) corner positions in the board-local frame (Z=0 plane).

        Derived from OpenCV's own chessboard corners rather than re-generated from the
        board dimensions, so the board has ONE geometry rather than two that agree until
        someone changes one of them.

        Recentred on the first interior corner because that is the frame this has always
        returned: OpenCV measures from the board's outer edge, so its first interior corner
        sits one square in. The offset is constant across every corner, so subtracting it
        reproduces the previous values exactly - asserted by
        `test_charuco_board_geometry.py`, which is what makes this a de-duplication rather
        than a change to where the calibration's world origin sits.
        """
        corners = np.asarray(self.cv2_board.getChessboardCorners(), dtype=np.float64)
        return corners - corners[0]

    @classmethod
    def create_test_data_7x5(cls) -> "CharucoBoardDefinition":
        return cls(squares_x=7, squares_y=5, square_length_mm=58.0)

    @classmethod
    def create_letter_size_5x3(cls) -> "CharucoBoardDefinition":
        return cls(squares_x=5, squares_y=3, square_length_mm=54.0)
