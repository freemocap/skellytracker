use anyhow::{bail, Result};
use serde::Deserialize;

/// ArUco DICT_4X4_250 — OpenCV enum value 0.
pub const ARUCO_DICT_4X4_250: i32 = 0;

fn default_marker_length_ratio() -> f64 {
    0.8
}

fn default_aruco_dictionary_id() -> i32 {
    ARUCO_DICT_4X4_250
}

/// Known charuco board geometry — fixed, never optimized.
///
/// Replaces Python's Pydantic `CharucoBoardDefinition`.
/// Validation lives in `new()` — an invalid board cannot be constructed.
#[derive(Debug, Clone, Deserialize)]
pub struct CharucoBoardDefinition {
    pub squares_x: u32,
    pub squares_y: u32,
    pub square_length_millimeters: f64,
    #[serde(default = "default_marker_length_ratio")]
    pub marker_length_ratio: f64,
    #[serde(default = "default_aruco_dictionary_id")]
    pub aruco_dictionary_id: i32,
}

impl Default for CharucoBoardDefinition {
    fn default() -> Self {
        Self {
            squares_x: 5,
            squares_y: 3,
            square_length_millimeters: 54.0,
            marker_length_ratio: 0.8,
            aruco_dictionary_id: ARUCO_DICT_4X4_250,
        }
    }
}

impl CharucoBoardDefinition {
    /// Validate and construct. Replaces Pydantic `@model_validator`.
    pub fn new(definition: Self) -> Result<Self> {
        let marker_length_millimeters =
            definition.marker_length_ratio * definition.square_length_millimeters;
        if marker_length_millimeters >= definition.square_length_millimeters {
            bail!(
                "marker_length ({}) must be < square_length ({})",
                marker_length_millimeters,
                definition.square_length_millimeters
            );
        }
        if definition.squares_x < 2 || definition.squares_y < 2 {
            bail!(
                "Board must have at least 2x2 squares, got {}x{}",
                definition.squares_x,
                definition.squares_y
            );
        }
        Ok(definition)
    }

    /// Computed: marker length in mm (replaces Pydantic `@computed_field`).
    pub fn aruco_marker_length_millimeters(&self) -> f64 {
        self.marker_length_ratio * self.square_length_millimeters
    }

    /// Number of internal chessboard corners.
    pub fn number_of_corners(&self) -> usize {
        ((self.squares_x - 1) * (self.squares_y - 1)) as usize
    }

    /// Corner positions in board-local frame, (number_of_corners, 3), Z=0.
    pub fn corner_positions_board_frame(&self) -> Vec<[f64; 3]> {
        let columns = self.squares_x - 1;
        let rows = self.squares_y - 1;
        let mut positions = Vec::with_capacity((columns * rows) as usize);
        for row in 0..rows {
            for column in 0..columns {
                positions.push([
                    column as f64 * self.square_length_millimeters,
                    row as f64 * self.square_length_millimeters,
                    0.0,
                ]);
            }
        }
        positions
    }

    /// Convenience constructor: 5x3 letter-size board.
    pub fn create_letter_size_5x3() -> Result<Self> {
        Self::new(Self {
            squares_x: 5,
            squares_y: 3,
            square_length_millimeters: 54.0,
            marker_length_ratio: 0.8,
            aruco_dictionary_id: ARUCO_DICT_4X4_250,
        })
    }

    /// Convenience constructor: 7x5 test board.
    pub fn create_test_data_7x5() -> Result<Self> {
        Self::new(Self {
            squares_x: 7,
            squares_y: 5,
            square_length_millimeters: 58.0,
            marker_length_ratio: 0.8,
            aruco_dictionary_id: ARUCO_DICT_4X4_250,
        })
    }
}
