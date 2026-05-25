use std::any::Any;

use crate::point_cloud::PointCloud;
use crate::traits::Observation;

/// 1:1 mirror of Python `CharucoObservation`.
///
/// **RULE #0 — DATA MODEL PARITY:** Every field in the Python
/// `CharucoObservation` MUST exist here with the same name and semantics.
/// Downstream consumers must NOT be able to tell whether an observation
/// came from the Rust or the Python backend.
///
/// Deferred fields (board pose, camera coords) are present but always
/// `None` until `solve_pnp` is implemented.

#[derive(Debug, Clone)]
pub struct CharucoObservation {
    pub tracker_type: &'static str,
    pub frame_number: u64,
    pub image_size: (u32, u32),

    /// Canonical: one row per charuco corner ID, NaN for undetected.
    pub points: PointCloud,

    // ── Board definition ──────────────────────────────────────────────
    pub all_charuco_ids: Vec<i32>,
    pub all_charuco_corners_in_object_coordinates: Option<Vec<[f64; 3]>>,
    pub all_aruco_ids: Vec<i32>,
    pub all_aruco_corners_in_object_coordinates: Option<Vec<[f64; 3]>>,

    // ── Raw detection data ────────────────────────────────────────────
    pub raw_charuco_corners: Option<Vec<[f64; 2]>>,
    pub detected_charuco_corner_ids: Vec<i32>,
    pub detected_charuco_corners_image_coordinates: Vec<[f64; 2]>,
    pub detected_charuco_corners_in_object_coordinates: Option<Vec<[f64; 3]>>,
    pub detected_aruco_marker_ids: Vec<i32>,
    pub detected_aruco_marker_corners: Vec<[[f64; 2]; 4]>,

    // ── Board pose (deferred — requires solve_pnp) ────────────────────
    pub charuco_board_translation_vector: Option<[f32; 3]>,
    pub charuco_board_rotation_vector: Option<[f32; 3]>,
    pub detected_charuco_corners_in_camera_coordinates: Option<Vec<[f64; 3]>>,
    pub detected_aruco_markers_in_camera_coordinates: Option<Vec<[f64; 3]>>,
}

impl CharucoObservation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frame_number: u64,
        points: PointCloud,
        image_size: (u32, u32),
        all_charuco_ids: Vec<i32>,
        all_charuco_corners_in_object_coordinates: Option<Vec<[f64; 3]>>,
        all_aruco_ids: Vec<i32>,
        all_aruco_corners_in_object_coordinates: Option<Vec<[f64; 3]>>,
        raw_charuco_corners: Option<Vec<[f64; 2]>>,
        detected_charuco_corner_ids: Vec<i32>,
        detected_charuco_corners_image_coordinates: Vec<[f64; 2]>,
        detected_charuco_corners_in_object_coordinates: Option<Vec<[f64; 3]>>,
        detected_aruco_marker_ids: Vec<i32>,
        detected_aruco_marker_corners: Vec<[[f64; 2]; 4]>,
    ) -> Self {
        CharucoObservation {
            tracker_type: "charuco_tracker",
            frame_number,
            points,
            image_size,
            all_charuco_ids,
            all_charuco_corners_in_object_coordinates,
            all_aruco_ids,
            all_aruco_corners_in_object_coordinates,
            raw_charuco_corners,
            detected_charuco_corner_ids,
            detected_charuco_corners_image_coordinates,
            detected_charuco_corners_in_object_coordinates,
            detected_aruco_marker_ids,
            detected_aruco_marker_corners,
            charuco_board_translation_vector: None,
            charuco_board_rotation_vector: None,
            detected_charuco_corners_in_camera_coordinates: None,
            detected_aruco_markers_in_camera_coordinates: None,
        }
    }

    // ── Python property equivalents ──────────────────────────────────

    pub fn charuco_empty(&self) -> bool {
        self.detected_charuco_corner_ids.is_empty()
    }

    pub fn aruco_empty(&self) -> bool {
        self.detected_aruco_marker_ids.is_empty()
    }
}

impl Observation for CharucoObservation {
    fn frame_number(&self) -> u64 {
        self.frame_number
    }

    fn point_cloud(&self) -> &PointCloud {
        &self.points
    }

    fn to_json(&self) -> String {
        let point_names: Vec<&str> = self.points.names.iter().map(|s| s.as_str()).collect();
        let xy = self.points.to_2d_array();
        let xy_json: Vec<Vec<f64>> = xy.outer_iter().map(|row| row.to_vec()).collect();
        let vis: Vec<f64> = self.points.visibility.to_vec();

        let aruco_corners_json: Vec<Vec<[f64; 2]>> = self
            .detected_aruco_marker_corners
            .iter()
            .map(|cs| cs.iter().copied().collect())
            .collect();

        serde_json::json!({
            "frame_number": self.frame_number,
            "tracker_type": self.tracker_type,
            "image_size": [self.image_size.0, self.image_size.1],
            "point_names": point_names,
            "xy": xy_json,
            "visibility": vis,
            "all_charuco_ids": self.all_charuco_ids,
            "all_charuco_corners_in_object_coordinates": self.all_charuco_corners_in_object_coordinates,
            "all_aruco_ids": self.all_aruco_ids,
            "all_aruco_corners_in_object_coordinates": self.all_aruco_corners_in_object_coordinates,
            "raw_charuco_corners": self.raw_charuco_corners,
            "detected_charuco_corner_ids": self.detected_charuco_corner_ids,
            "detected_charuco_corners_image_coordinates": self.detected_charuco_corners_image_coordinates,
            "detected_charuco_corners_in_object_coordinates": self.detected_charuco_corners_in_object_coordinates,
            "detected_aruco_marker_ids": self.detected_aruco_marker_ids,
            "detected_aruco_marker_corners": aruco_corners_json,
        })
        .to_string()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
