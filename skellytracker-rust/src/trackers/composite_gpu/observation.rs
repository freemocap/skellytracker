//! CompositeGPUObservation — 165-point hybrid detection result.
//!
//! Composition (from rtmo_hybrid.yaml):
//!   body(0..17) + right_hand(17..38) + left_hand(38..59) + face(59..165)
//!
//! Rule #0: Every field in Python's CompositeGPUObservation MUST exist here.

use std::any::Any;
use std::sync::LazyLock;

use ndarray::{Array2, Array3};
use crate::point_cloud::PointCloud;
use crate::traits::Observation;

// ---------------------------------------------------------------------------
// Point names
// ---------------------------------------------------------------------------

const BODY_NAMES: &[&str] = &[
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
];

const HAND_NAMES: &[&str] = &[
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
];

const FACE_NAMES: &[&str] = &[
    "face_0000", "face_0001", "face_0002", "face_0003", "face_0004",
    "face_0005", "face_0006", "face_0007", "face_0008", "face_0009",
    "face_0010", "face_0011", "face_0012", "face_0013", "face_0014",
    "face_0015", "face_0016", "face_0017", "face_0018", "face_0019",
    "face_0020", "face_0021", "face_0022", "face_0023", "face_0024",
    "face_0025", "face_0026", "face_0027", "face_0028", "face_0029",
    "face_0030", "face_0031", "face_0032", "face_0033", "face_0034",
    "face_0035", "face_0036", "face_0037", "face_0038", "face_0039",
    "face_0040", "face_0041", "face_0042", "face_0043", "face_0044",
    "face_0045", "face_0046", "face_0047", "face_0048", "face_0049",
    "face_0050", "face_0051", "face_0052", "face_0053", "face_0054",
    "face_0055", "face_0056", "face_0057", "face_0058", "face_0059",
    "face_0060", "face_0061", "face_0062", "face_0063", "face_0064",
    "face_0065", "face_0066", "face_0067", "face_0068", "face_0069",
    "face_0070", "face_0071", "face_0072", "face_0073", "face_0074",
    "face_0075", "face_0076", "face_0077", "face_0078", "face_0079",
    "face_0080", "face_0081", "face_0082", "face_0083", "face_0084",
    "face_0085", "face_0086", "face_0087", "face_0088", "face_0089",
    "face_0090", "face_0091", "face_0092", "face_0093", "face_0094",
    "face_0095", "face_0096", "face_0097", "face_0098", "face_0099",
    "face_0100", "face_0101", "face_0102", "face_0103", "face_0104", "face_0105",
];

pub const NUM_BODY: usize = 17;
pub const NUM_HAND: usize = 21;
pub const NUM_FACE: usize = 106;
pub const NUM_HYBRID: usize = NUM_BODY + 2 * NUM_HAND + NUM_FACE; // 165

static HYBRID_NAMES: LazyLock<Vec<String>> = LazyLock::new(|| {
    let mut n = Vec::with_capacity(NUM_HYBRID);
    n.extend(BODY_NAMES.iter().map(|s| s.to_string()));
    n.extend(HAND_NAMES.iter().map(|s| format!("right_hand_{s}")));
    n.extend(HAND_NAMES.iter().map(|s| format!("left_hand_{s}")));
    n.extend(FACE_NAMES.iter().map(|s| s.to_string()));
    n
});

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CompositeGpuObservation {
    pub tracker_type: &'static str,
    pub frame_number: u64,
    pub image_size: (u32, u32),
    pub points: PointCloud,
    pub body_keypoints: Array3<f64>,
    pub body_scores: Array2<f32>,
    pub hands_keypoints: Array3<f64>,
    pub hands_scores: Array2<f32>,
    pub face_keypoints: Array3<f64>,
    pub face_scores: Array2<f32>,
}

impl CompositeGpuObservation {
    pub fn from_detection_results(
        frame_number: u64,
        image_size: (u32, u32),
        body_keypoints: Array3<f64>,
        body_scores: Array2<f32>,
        hands_keypoints: Array3<f64>,
        hands_scores: Array2<f32>,
        face_keypoints: Array3<f64>,
        face_scores: Array2<f32>,
    ) -> Self {
        let n = NUM_HYBRID;
        let mut xyz = Array2::from_elem((n, 3), f64::NAN);
        let mut visibility = ndarray::Array1::zeros(n);

        // Body: first person
        if body_keypoints.shape()[0] > 0 && body_keypoints.shape()[1] >= 17 {
            for k in 0..17usize {
                xyz[[k, 0]] = body_keypoints[[0, k, 0]];
                xyz[[k, 1]] = body_keypoints[[0, k, 1]];
                visibility[k] = body_scores[[0, k]] as f64;
            }
        }

        // Hands: right(0:21) then left(21:42)
        if hands_keypoints.shape()[0] > 0 && hands_keypoints.shape()[1] >= 42 {
            for k in 0..21 {
                xyz[[NUM_BODY + k, 0]] = hands_keypoints[[0, k, 0]];
                xyz[[NUM_BODY + k, 1]] = hands_keypoints[[0, k, 1]];
                visibility[NUM_BODY + k] = hands_scores[[0, k]] as f64;
            }
            for k in 0..21 {
                xyz[[NUM_BODY + NUM_HAND + k, 0]] = hands_keypoints[[0, 21 + k, 0]];
                xyz[[NUM_BODY + NUM_HAND + k, 1]] = hands_keypoints[[0, 21 + k, 1]];
                visibility[NUM_BODY + NUM_HAND + k] = hands_scores[[0, 21 + k]] as f64;
            }
        }

        // Face (106)
        if face_keypoints.shape()[0] > 0 && face_keypoints.shape()[1] >= 106 {
            let face_start = NUM_BODY + 2 * NUM_HAND;
            for k in 0..106 {
                xyz[[face_start + k, 0]] = face_keypoints[[0, k, 0]];
                xyz[[face_start + k, 1]] = face_keypoints[[0, k, 1]];
                visibility[face_start + k] = face_scores[[0, k]] as f64;
            }
        }

        Self {
            tracker_type: "rtmo_hybrid",
            frame_number,
            image_size,
            points: PointCloud::new(HYBRID_NAMES.to_vec(), xyz, visibility),
            body_keypoints,
            body_scores,
            hands_keypoints,
            hands_scores,
            face_keypoints,
            face_scores,
        }
    }

    pub fn empty(frame_number: u64, image_size: (u32, u32)) -> Self {
        Self::from_detection_results(
            frame_number, image_size,
            Array3::from_elem((0, 17, 2), 0.0),
            Array2::from_elem((0, 17), 0.0),
            Array3::from_elem((0, 42, 2), 0.0),
            Array2::from_elem((0, 42), 0.0),
            Array3::from_elem((0, 106, 2), 0.0),
            Array2::from_elem((0, 106), 0.0),
        )
    }
}

impl Observation for CompositeGpuObservation {
    fn frame_number(&self) -> u64 { self.frame_number }
    fn point_cloud(&self) -> &PointCloud { &self.points }

    fn to_json(&self) -> String {
        let names: Vec<&str> = self.points.names.iter().map(|s| s.as_str()).collect();
        let xy = self.points.to_2d_array();
        let xy_json: Vec<Vec<f64>> = xy.outer_iter().map(|r| r.to_vec()).collect();
        let vis: Vec<f64> = self.points.visibility.to_vec();
        serde_json::json!({
            "tracker_type": self.tracker_type,
            "frame_number": self.frame_number,
            "image_size": [self.image_size.0, self.image_size.1],
            "point_names": names,
            "xy": xy_json,
            "visibility": vis,
        }).to_string()
    }

    fn as_any(&self) -> &dyn Any { self }
}
