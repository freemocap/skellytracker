//! RTMPoseObservation — 133-keypoint wholebody pose detection result.
//!
//! Rule #0: Every field in Python's RTMPoseObservation MUST exist here
//! with the same name, same type, and same semantics.

use std::any::Any;

use ndarray::{Array2, Array3};
use ndarray::Axis;
use crate::point_cloud::PointCloud;
use crate::traits::Observation;

// ---------------------------------------------------------------------------
// 133 point names in schema composition order:
//   body(0..22) + right_hand(23..43) + left_hand(44..64) + face(65..132)
// ---------------------------------------------------------------------------

const BODY_NAMES: &[&str] = &[
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel", "left_big_toe",
    "right_big_toe", "left_small_toe", "right_small_toe",
];

const HAND_NAMES: &[&str] = &[
    "root", "thumb1", "thumb2", "thumb3", "thumb4",
    "forefinger1", "forefinger2", "forefinger3", "forefinger4",
    "middle_finger1", "middle_finger2", "middle_finger3", "middle_finger4",
    "ring_finger1", "ring_finger2", "ring_finger3", "ring_finger4",
    "pinky_finger1", "pinky_finger2", "pinky_finger3", "pinky_finger4",
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
    "face_0065", "face_0066", "face_0067",
];

/// Build the 133 element name list in schema order.
pub fn rtmpose_names() -> Vec<String> {
    let mut names: Vec<String> = Vec::with_capacity(133);
    names.extend(BODY_NAMES.iter().map(|s| s.to_string()));
    names.extend(HAND_NAMES.iter().map(|s| format!("right_hand_{s}")));
    names.extend(HAND_NAMES.iter().map(|s| format!("left_hand_{s}")));
    names.extend(FACE_NAMES.iter().map(|s| s.to_string()));
    names
}

// ---------------------------------------------------------------------------
// Permutation: rtmlib native → schema order
//
// rtmlib COCO-WholeBody: body(0..22) + face(23..90) + left_hand(91..111) + right_hand(112..132)
// Schema composition:    body(0..22) + right_hand(23..43) + left_hand(44..64) + face(65..132)
// ---------------------------------------------------------------------------

pub fn rtmlib_to_schema_perm() -> Vec<usize> {
    let mut perm: Vec<usize> = Vec::with_capacity(133);
    perm.extend(0..23);         // body stays in place
    perm.extend(112..133);       // right_hand moves up
    perm.extend(91..112);        // left_hand moves up
    perm.extend(23..91);         // face moves down
    perm
}

// ---------------------------------------------------------------------------
// RTMPoseObservation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RtmPoseObservation {
    pub tracker_type: &'static str,
    pub frame_number: u64,
    pub image_size: (u32, u32),
    /// PointCloud with 133 points in SCHEMA order (body + right_hand + left_hand + face).
    pub points: PointCloud,
    /// Raw keypoints in rtmlib's NATIVE order: shape (num_persons, 133, 2) float64.
    pub keypoints: Array3<f64>,
    /// Raw scores in rtmlib's NATIVE order: shape (num_persons, 133) float32.
    pub scores: Array2<f32>,
    /// YOLOX person bounding box [x1, y1, x2, y2] in image coords (first person only).
    /// Not in Python's observation — added for Rust annotation.
    pub person_bbox: Option<[f64; 4]>,
}

impl RtmPoseObservation {
    /// Construct from detection results (matching Python's `from_detection_results`).
    ///
    /// `keypoints`: (num_persons, 133, 2) float64 in rtmlib native order
    /// `scores`: (num_persons, 133) float32 in rtmlib native order
    pub fn from_detection_results(
        frame_number: u64,
        keypoints: Array3<f64>,
        scores: Array2<f32>,
        image_size: (u32, u32),
    ) -> Self {
        let names = rtmpose_names();
        let perm = rtmlib_to_schema_perm();
        let n = names.len(); // 133

        let (points_2d, confidence) = if keypoints.shape()[0] > 0 {
            // Take first detected person, permute to schema order
            let kp_slice = keypoints.slice(ndarray::s![0, .., ..]); // (133, 2)
            let sc_slice = scores.slice(ndarray::s![0, ..]);        // (133,)

            let mut points_2d = Array2::<f64>::zeros((n, 2));
            let mut confidence = vec![0.0_f64; n];
            for (target_idx, &src_idx) in perm.iter().enumerate() {
                if src_idx < 133 {
                    points_2d[[target_idx, 0]] = kp_slice[[src_idx, 0]];
                    points_2d[[target_idx, 1]] = kp_slice[[src_idx, 1]];
                    confidence[target_idx] = sc_slice[src_idx] as f64;
                }
            }
            (points_2d, ndarray::Array1::from_vec(confidence))
        } else {
            // No persons detected — all NaN / zero visibility
            let points_2d = Array2::from_elem((n, 2), f64::NAN);
            let confidence = ndarray::Array1::zeros(n);
            (points_2d, confidence)
        };

        // Build xyz: (N, 3) with z=0
        let mut xyz = Array2::from_elem((n, 3), f64::NAN);
        for i in 0..n {
            xyz[[i, 0]] = points_2d[[i, 0]];
            xyz[[i, 1]] = points_2d[[i, 1]];
            xyz[[i, 2]] = 0.0;
        }

        let cloud = PointCloud::new(names, xyz, confidence);

        Self {
            tracker_type: "rtmpose",
            frame_number,
            image_size,
            points: cloud,
            keypoints,
            scores,
            person_bbox: None,
        }
    }

    /// Create an empty observation (no detection).
    pub fn empty(frame_number: u64, image_size: (u32, u32)) -> Self {
        let n = 133;
        let keypoints = Array3::from_elem((0, n, 2), 0.0_f64);
        let scores = Array2::from_elem((0, n), 0.0_f32);
        Self::from_detection_results(frame_number, keypoints, scores, image_size)
    }
}

impl Observation for RtmPoseObservation {
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

        // Raw keypoints/scores in rtmlib native order
        let kp_shape = self.keypoints.shape();
        let kp_json: Vec<Vec<Vec<f64>>> = self
            .keypoints
            .axis_iter(Axis(0))
            .map(|person| {
                person
                    .axis_iter(Axis(0))
                    .map(|row| row.to_vec())
                    .collect()
            })
            .collect();

        let sc_json: Vec<Vec<f32>> = self
            .scores
            .axis_iter(Axis(0))
            .map(|row| row.to_vec())
            .collect();

        serde_json::json!({
            "tracker_type": self.tracker_type,
            "frame_number": self.frame_number,
            "image_size": [self.image_size.0, self.image_size.1],
            "point_names": point_names,
            "xy": xy_json,
            "visibility": vis,
            "keypoints": kp_json,
            "scores": sc_json,
        })
        .to_string()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
