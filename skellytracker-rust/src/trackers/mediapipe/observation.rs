//! MediaPipeObservation — 211-point holistic detection result.
//!
//! Composition (from mediapipe_holistic.yaml):
//!   body(0..33) + right_hand(33..54) + left_hand(54..75) + face_contour(75..211)
//!
//! Rule #0: Every field in Python's MediapipeCompositeObservation MUST exist here
//! with the same name, same type, and same semantics.

use std::any::Any;
use std::sync::LazyLock;

use ndarray::Array2;
use crate::point_cloud::PointCloud;
use crate::traits::Observation;

// ---------------------------------------------------------------------------
// Point names in holistic YAML composition order
// ---------------------------------------------------------------------------

const BODY_NAMES: &[&str] = &[
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
];

const HAND_NAMES: &[&str] = &[
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
];

const FACE_CONTOUR_NAMES: &[&str] = &[
    "face_0000", "face_0007", "face_0010", "face_0013", "face_0014",
    "face_0017", "face_0021", "face_0033", "face_0037", "face_0039",
    "face_0040", "face_0046", "face_0052", "face_0053", "face_0054",
    "face_0055", "face_0058", "face_0061", "face_0063", "face_0065",
    "face_0066", "face_0067", "face_0070", "face_0078", "face_0080",
    "face_0081", "face_0082", "face_0084", "face_0087", "face_0088",
    "face_0091", "face_0093", "face_0095", "face_0103", "face_0105",
    "face_0107", "face_0109", "face_0127", "face_0132", "face_0133",
    "face_0136", "face_0144", "face_0145", "face_0146", "face_0148",
    "face_0149", "face_0150", "face_0152", "face_0153", "face_0154",
    "face_0155", "face_0157", "face_0158", "face_0159", "face_0160",
    "face_0161", "face_0162", "face_0163", "face_0172", "face_0173",
    "face_0176", "face_0178", "face_0181", "face_0185", "face_0191",
    "face_0234", "face_0246", "face_0249", "face_0251", "face_0263",
    "face_0267", "face_0269", "face_0270", "face_0276", "face_0282",
    "face_0283", "face_0284", "face_0285", "face_0288", "face_0291",
    "face_0293", "face_0295", "face_0296", "face_0297", "face_0300",
    "face_0308", "face_0310", "face_0311", "face_0312", "face_0314",
    "face_0317", "face_0318", "face_0321", "face_0323", "face_0324",
    "face_0332", "face_0334", "face_0336", "face_0338", "face_0356",
    "face_0361", "face_0362", "face_0365", "face_0373", "face_0374",
    "face_0375", "face_0377", "face_0378", "face_0379", "face_0380",
    "face_0381", "face_0382", "face_0384", "face_0385", "face_0386",
    "face_0387", "face_0388", "face_0389", "face_0390", "face_0397",
    "face_0398", "face_0400", "face_0402", "face_0405", "face_0409",
    "face_0415", "face_0454", "face_0466", "face_0469", "face_0470",
    "face_0471", "face_0472", "face_0474", "face_0475", "face_0476",
    "face_0477",
];

pub const NUM_POSE_LANDMARKS: usize = 33;
pub const NUM_HAND_LANDMARKS: usize = 21;
pub const NUM_FACE_CONTOUR_LANDMARKS: usize = 136;
pub const NUM_HOLISTIC_LANDMARKS: usize = NUM_POSE_LANDMARKS + 2 * NUM_HAND_LANDMARKS + NUM_FACE_CONTOUR_LANDMARKS; // 211

static HOLISTIC_NAMES: LazyLock<Vec<String>> = LazyLock::new(|| {
    let mut names: Vec<String> = Vec::with_capacity(NUM_HOLISTIC_LANDMARKS);
    names.extend(BODY_NAMES.iter().map(|s| s.to_string()));
    names.extend(HAND_NAMES.iter().map(|s| format!("right_hand_{s}")));
    names.extend(HAND_NAMES.iter().map(|s| format!("left_hand_{s}")));
    names.extend(FACE_CONTOUR_NAMES.iter().map(|s| s.to_string()));
    names
});

// ---------------------------------------------------------------------------
// Slice boundaries for sub-component views
// ---------------------------------------------------------------------------

pub const BODY_START: usize = 0;
pub const BODY_END: usize = NUM_POSE_LANDMARKS;
pub const RHAND_START: usize = BODY_END;
pub const RHAND_END: usize = RHAND_START + NUM_HAND_LANDMARKS;
pub const LHAND_START: usize = RHAND_END;
pub const LHAND_END: usize = LHAND_START + NUM_HAND_LANDMARKS;
pub const FACE_START: usize = LHAND_END;
pub const FACE_END: usize = FACE_START + NUM_FACE_CONTOUR_LANDMARKS;

// ---------------------------------------------------------------------------
// MediaPipeObservation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MediaPipeObservation {
    pub tracker_type: &'static str,
    pub frame_number: u64,
    pub image_size: (u32, u32),
    /// 211-point fused PointCloud in holistic YAML order.
    pub points: PointCloud,
    /// Detection state flags (needed for downstream consumers).
    pub has_pose: bool,
    pub has_right_hand: bool,
    pub has_left_hand: bool,
    pub has_face: bool,
}

impl MediaPipeObservation {
    /// Build from extracted Python detection data.
    pub fn build(
        frame_number: u64,
        image_size: (u32, u32),
        names: Vec<String>,
        xyz: Array2<f64>,
        visibility: ndarray::Array1<f64>,
        has_pose: bool,
        has_right_hand: bool,
        has_left_hand: bool,
        has_face: bool,
    ) -> Self {
        let cloud = PointCloud::new(names, xyz, visibility);
        Self {
            tracker_type: "mediapipe_composite",
            frame_number,
            image_size,
            points: cloud,
            has_pose,
            has_right_hand,
            has_left_hand,
            has_face,
        }
    }

    /// Empty observation (no detections).
    pub fn empty(frame_number: u64, image_size: (u32, u32)) -> Self {
        let n = NUM_HOLISTIC_LANDMARKS;
        let xyz = Array2::from_elem((n, 3), f64::NAN);
        let vis = ndarray::Array1::zeros(n);
        Self::build(
            frame_number,
            image_size,
            HOLISTIC_NAMES.to_vec(),
            xyz,
            vis,
            false,
            false,
            false,
            false,
        )
    }
}

impl Observation for MediaPipeObservation {
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

        serde_json::json!({
            "tracker_type": self.tracker_type,
            "frame_number": self.frame_number,
            "image_size": [self.image_size.0, self.image_size.1],
            "point_names": point_names,
            "xy": xy_json,
            "visibility": vis,
            "has_pose": self.has_pose,
            "has_right_hand": self.has_right_hand,
            "has_left_hand": self.has_left_hand,
            "has_face": self.has_face,
        })
        .to_string()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
