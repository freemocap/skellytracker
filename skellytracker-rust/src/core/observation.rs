use super::point_cloud::PointCloud;

/// Top-level observation. Every detection produces one.
#[derive(Debug, Clone)]
pub struct Observation {
    pub frame_number: u64,
    pub tracker_kind: TrackerKind,
    pub points: PointCloud,
    pub payload: ObservationPayload,
}

/// Which tracker produced this observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackerKind {
    Charuco,
    // Future: MediaPipe, RtmPose, CompositeGpu,
}

/// Tracker-specific detection data.
#[derive(Debug, Clone)]
pub enum ObservationPayload {
    Charuco {
        all_charuco_ids: Vec<i32>,
        all_aruco_ids: Vec<i32>,
        detected_charuco_corner_ids: Option<Vec<i32>>,
        detected_charuco_corners: Option<Vec<[f64; 2]>>,
        detected_aruco_marker_ids: Option<Vec<i32>>,
        detected_aruco_marker_corners: Option<Vec<[[f64; 2]; 4]>>,
        board_rotation_vector: Option<[f64; 3]>,
        board_translation_vector: Option<[f64; 3]>,
        detected_charuco_corners_in_camera_coordinates: Option<Vec<[f64; 3]>>,
    },
}
