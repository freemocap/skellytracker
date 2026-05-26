//! CompositeGPU tracker — multi-model GPU pipeline.
//!
//! Phase 3 (bridge): tracker struct, Trait impl, draw_markers_into.

pub mod observation;
pub mod roi;
pub mod session;

use ndarray::{Array2, Array3};
use opencv::core::{Point, Scalar};
use opencv::imgproc;
use opencv::prelude::*;

use crate::onnx_utils::session_builder::Provider;
use crate::trackers::composite_gpu::observation::{CompositeGpuObservation, NUM_BODY, NUM_HAND, NUM_HYBRID};
use crate::trackers::composite_gpu::session::{CompositeGpuSession, CompositeGpuSessionConfig, HandResult};
use crate::traits::{Observation, Tracker};

// ---------------------------------------------------------------------------
// Skeleton connections (hardcoded from YAML definitions)
// ---------------------------------------------------------------------------

/// Body skeleton edges (16 connections from rtmo_body_17.yaml), 0-indexed.
const BODY_LINKS: &[(usize, usize)] = &[
    (0, 1), (1, 3), (0, 2), (2, 4),   // nose → eyes → ears
    (5, 6),                             // left_shoulder ↔ right_shoulder
    (5, 7), (7, 9),                    // left_shoulder → left_elbow → left_wrist
    (6, 8), (8, 10),                   // right_shoulder → right_elbow → right_wrist
    (5, 11), (6, 12),                  // shoulders → hips
    (11, 12),                           // left_hip ↔ right_hip
    (11, 13), (13, 15),                // left_hip → left_knee → left_ankle
    (12, 14), (14, 16),                // right_hip → right_knee → right_ankle
];

/// Per-hand skeleton edges (23 connections from mediapipe_hand.yaml), local 0..20.
const HAND_LINKS: &[(usize, usize)] = &[
    (0, 1), (1, 2), (2, 3), (3, 4),   // wrist → thumb chain
    (0, 5), (5, 6), (6, 7), (7, 8),   // wrist → index chain
    (0, 9), (9, 10), (10, 11), (11, 12), // wrist → middle chain
    (0, 13), (13, 14), (14, 15), (15, 16), // wrist → ring chain
    (0, 17), (17, 18), (18, 19), (19, 20), // wrist → pinky chain
    (5, 9), (9, 13), (13, 17),         // palm arches
];

// ---------------------------------------------------------------------------
// Colors (BGR)
// ---------------------------------------------------------------------------

const BODY_COLOR: Scalar = Scalar::new(0.0, 255.0, 0.0, 0.0);       // green
const RIGHT_HAND_COLOR: Scalar = Scalar::new(0.0, 0.0, 255.0, 0.0);  // red
const LEFT_HAND_COLOR: Scalar = Scalar::new(255.0, 0.0, 0.0, 0.0);   // blue
const FACE_COLOR: Scalar = Scalar::new(0.0, 255.0, 255.0, 0.0);      // yellow
const ROI_COLOR: Scalar = Scalar::new(200.0, 200.0, 200.0, 0.0);     // light gray

const KPT_THRESHOLD: f64 = 0.3;
const KPT_RADIUS: i32 = 2;
const LINE_WIDTH: i32 = 2;

// ---------------------------------------------------------------------------
// Tracker
// ---------------------------------------------------------------------------

pub struct CompositeGpuTracker {
    pub preset: String,
    session: CompositeGpuSession,
}

impl CompositeGpuTracker {
    pub fn new(mode: &str, provider: Provider) -> Result<Self, Box<dyn std::error::Error>> {
        let preset = crate::onnx_utils::model_registry::TrackerPreset::Medium; // default
        let cfg = CompositeGpuSession::preset(preset);
        let cfg = CompositeGpuSessionConfig {
            execution_provider: provider,
            ..cfg
        };

        let session = CompositeGpuSession::create(cfg)?;
        Ok(Self { preset: mode.to_string(), session })
    }

    pub fn detect(&mut self, frame_number: u64, image: &Mat) -> CompositeGpuObservation {
        let image_size = (image.rows() as u32, image.cols() as u32);

        let (body, hands, face) = self.session.predict(image);

        // Assemble keypoint arrays
        let body_kpts = body.keypoints;   // (n_persons, 17, 2)
        let body_sc = body.scores;        // (n_persons, 17)

        // Concatenate right+left hands into (1, 42, 2)
        let (hands_kpts, hands_sc) = assemble_hands_arrays(&hands);

        // Face: (1, 106, 2)
        let face_kpts_arr = face.keypoints.insert_axis(ndarray::Axis(0)); // (1, 106, 2)
        let face_sc_arr = face.scores;

        CompositeGpuObservation::from_detection_results(
            frame_number, image_size,
            body_kpts, body_sc,
            hands_kpts, hands_sc,
            face_kpts_arr, face_sc_arr,
        )
    }

    pub fn draw_markers_into(&self, image: &mut Mat, obs: &dyn Observation) {
        let o = match obs.as_any().downcast_ref::<CompositeGpuObservation>() {
            Some(c) => c,
            None => return,
        };

        if o.points.names.is_empty() { return; }

        let xy = o.points.to_2d_array(); // (165, 2)
        let vis = &o.points.visibility;   // (165,)

        // Skeleton lines
        let body_end = NUM_BODY;
        let rhand_end = NUM_BODY + NUM_HAND;
        let lhand_end = NUM_BODY + 2 * NUM_HAND;

        // Body connections
        draw_links(image, &xy, vis, BODY_LINKS, 0, BODY_COLOR);

        // Right hand connections
        let rhand_links: Vec<(usize, usize)> = HAND_LINKS.iter()
            .map(|&(a, b)| (a + body_end, b + body_end))
            .collect();
        draw_links(image, &xy, vis, &rhand_links, 0, RIGHT_HAND_COLOR);

        // Left hand connections
        let lhand_links: Vec<(usize, usize)> = HAND_LINKS.iter()
            .map(|&(a, b)| (a + rhand_end, b + rhand_end))
            .collect();
        draw_links(image, &xy, vis, &lhand_links, 0, LEFT_HAND_COLOR);

        // Face keypoints only (no skeleton lines — too many)
        for i in lhand_end..NUM_HYBRID {
            if vis[i] < KPT_THRESHOLD { continue; }
            let px = xy[[i, 0]] as i32;
            let py = xy[[i, 1]] as i32;
            if px == 0 && py == 0 { continue; }
            let _ = imgproc::circle(image, Point::new(px, py), 1, FACE_COLOR, -1, imgproc::LINE_8, 0);
        }

        // Body keypoints (larger circles)
        for i in 0..body_end {
            if vis[i] < KPT_THRESHOLD { continue; }
            let px = xy[[i, 0]] as i32;
            let py = xy[[i, 1]] as i32;
            if px == 0 && py == 0 { continue; }
            let _ = imgproc::circle(image, Point::new(px, py), KPT_RADIUS, BODY_COLOR, -1, imgproc::LINE_8, 0);
        }

        // Hand keypoints
        for i in body_end..lhand_end {
            if vis[i] < KPT_THRESHOLD { continue; }
            let px = xy[[i, 0]] as i32;
            let py = xy[[i, 1]] as i32;
            if px == 0 && py == 0 { continue; }
            let color = if i < rhand_end { RIGHT_HAND_COLOR } else { LEFT_HAND_COLOR };
            let _ = imgproc::circle(image, Point::new(px, py), KPT_RADIUS, color, -1, imgproc::LINE_8, 0);
        }

        // Wrist highlight circles (body index 9=left_wrist, 10=right_wrist)
        for (wrist_idx, label) in [(9, "L"), (10, "R")] {
            if wrist_idx < body_end && vis[wrist_idx] >= KPT_THRESHOLD {
                let wx = xy[[wrist_idx, 0]] as i32;
                let wy = xy[[wrist_idx, 1]] as i32;
                if wx > 0 || wy > 0 {
                    let color = if wrist_idx == 9 { LEFT_HAND_COLOR } else { RIGHT_HAND_COLOR };
                    let _ = imgproc::circle(image, Point::new(wx, wy), 8, color, 2, imgproc::LINE_8, 0);
                    let _ = imgproc::circle(image, Point::new(wx, wy), 3, Scalar::new(255.0, 255.0, 255.0, 0.0), -1, imgproc::LINE_8, 0);
                    let _ = imgproc::put_text(image, label, Point::new(wx + 10, wy),
                        imgproc::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, imgproc::LINE_8, false);
                }
            }
        }
    }
}

impl Tracker for CompositeGpuTracker {
    fn process_image(&mut self, frame_number: u64, image: &Mat) -> Box<dyn Observation> {
        Box::new(self.detect(frame_number, image))
    }

    fn annotate_image(&self, image: &Mat, obs: &dyn Observation) -> Mat {
        let mut annotated = image.clone();
        self.draw_markers_into(&mut annotated, obs);
        annotated
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn draw_links(
    image: &mut Mat,
    xy: &Array2<f64>,
    vis: &ndarray::Array1<f64>,
    links: &[(usize, usize)],
    _offset: usize,
    color: Scalar,
) {
    for &(a, b) in links {
        if a >= xy.nrows() || b >= xy.nrows() { continue; }
        if vis[a] < KPT_THRESHOLD || vis[b] < KPT_THRESHOLD { continue; }
        let x0 = xy[[a, 0]]; let y0 = xy[[a, 1]];
        let x1 = xy[[b, 0]]; let y1 = xy[[b, 1]];
        if x0.is_nan() || y0.is_nan() || x1.is_nan() || y1.is_nan() { continue; }
        let p0 = Point::new(x0 as i32, y0 as i32);
        let p1 = Point::new(x1 as i32, y1 as i32);
        if p0.x == 0 && p0.y == 0 && p1.x == 0 && p1.y == 0 { continue; }
        let _ = imgproc::line(image, p0, p1, color, LINE_WIDTH, imgproc::LINE_8, 0);
    }
}

/// Assemble right + left hand keypoints into (1, 42, 2) array.
fn assemble_hands_arrays(hands: &HandResult) -> (Array3<f64>, Array2<f32>) {
    let mut kpts = Array3::<f64>::zeros((1, 42, 2));
    let mut scores = Array2::<f32>::zeros((1, 42));
    for k in 0..NUM_HAND {
        kpts[[0, k, 0]] = hands.right_keypoints[[k, 0]];
        kpts[[0, k, 1]] = hands.right_keypoints[[k, 1]];
        scores[[0, k]] = hands.right_scores.get(k).copied().unwrap_or(0.0);
    }
    for k in 0..NUM_HAND {
        kpts[[0, NUM_HAND + k, 0]] = hands.left_keypoints[[k, 0]];
        kpts[[0, NUM_HAND + k, 1]] = hands.left_keypoints[[k, 1]];
        scores[[0, NUM_HAND + k]] = hands.left_scores.get(k).copied().unwrap_or(0.0);
    }
    (kpts, scores)
}
