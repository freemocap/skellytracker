import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, computed_field, model_validator


class CharucoBoardDefinition(BaseModel):
    """Known charuco board geometry — fixed, never optimized.

    Single source of truth for board parameters. Both the detection tracker
    and calibration solver construct their cv2/solver-specific board
    representations from this.
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
    def n_corners(self) -> int:
        """Number of internal corners (used as calibration input points)."""
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def corner_positions_board_frame(self) -> NDArray[np.float64]:
        """(n_corners, 3) corner positions in the board-local frame (Z=0 plane)."""
        cols = self.squares_x - 1
        rows = self.squares_y - 1
        objp = np.zeros((cols * rows, 3), dtype=np.float64)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= self.square_length_mm
        return objp

    @classmethod
    def create_test_data_7x5(cls) -> "CharucoBoardDefinition":
        return cls(squares_x=7, squares_y=5, square_length_mm=58.0)

    @classmethod
    def create_letter_size_5x3(cls) -> "CharucoBoardDefinition":
        return cls(squares_x=5, squares_y=3, square_length_mm=54.0)
