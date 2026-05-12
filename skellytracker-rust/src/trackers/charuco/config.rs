use serde::Deserialize;

use super::board::CharucoBoardDefinition;

/// Configuration for the Charuco detector.
#[derive(Debug, Clone, Deserialize)]
pub struct CharucoDetectorConfig {
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,
    #[serde(default)]
    pub board_definition: CharucoBoardDefinition,
}

fn default_confidence_threshold() -> f64 {
    0.5
}

impl Default for CharucoDetectorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            board_definition: CharucoBoardDefinition::create_letter_size_5x3()
                .expect("default board definition must be valid"),
        }
    }
}

/// Configuration for the Charuco image annotator.
#[derive(Debug, Clone)]
pub struct CharucoAnnotatorConfig {
    pub show_tracks: Option<usize>,
    pub corner_marker_size: i32,
    pub corner_marker_thickness: i32,
    pub corner_marker_color: (u8, u8, u8),
    pub aruco_lines_thickness: i32,
    pub aruco_lines_color: (u8, u8, u8),
    pub text_color: (u8, u8, u8),
    pub text_size: f64,
    pub text_thickness: i32,
    pub show_overlay: bool,
}

impl Default for CharucoAnnotatorConfig {
    fn default() -> Self {
        Self {
            show_tracks: Some(15),
            corner_marker_size: 10,
            corner_marker_thickness: 2,
            corner_marker_color: (255, 0, 255),
            aruco_lines_thickness: 2,
            aruco_lines_color: (0, 255, 0),
            text_color: (215, 115, 40),
            text_size: 0.5,
            text_thickness: 2,
            show_overlay: false,
        }
    }
}
