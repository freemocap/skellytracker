//! Stateless ROI crop utilities ported from Python's `roi_crop_utils.py`.
//!
//! Pure geometry — no I/O, no ML. Works with any body keypoint source as long
//! as the caller provides (x, y) coordinates in image pixel space.
//!
//! State management (last hand sizes, smoothed ROI centers/sizes, handedness
//! tracking) belongs in the caller. These functions are pure math.

use ndarray::ArrayView2;

// ---------------------------------------------------------------------------
// ROIBox
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct RoiBox {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl RoiBox {
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self { x, y, width, height }
    }
}

// ---------------------------------------------------------------------------
// Square ROI computation
// ---------------------------------------------------------------------------

/// Compute a square ROI clamped to image bounds.
pub fn compute_square_roi(center_x: i32, center_y: i32, size: i32, image_w: i32, image_h: i32) -> RoiBox {
    let half = size / 2;
    let x = (center_x - half).max(0);
    let y = (center_y - half).max(0);
    let x2 = (center_x + half).min(image_w);
    let y2 = (center_y + half).min(image_h);
    RoiBox::new(x, y, x2 - x, y2 - y)
}

// ---------------------------------------------------------------------------
// EMA smoothing
// ---------------------------------------------------------------------------

/// Apply exponential moving average smoothing to ROI center and size.
///
/// Returns `(raw_cx, raw_cy, raw_size)` on cold start (prev = None).
/// `alpha = 0.0` means no smoothing (use raw); `alpha = 1.0` means frozen.
pub fn smooth_roi_params(
    raw_cx: f64,
    raw_cy: f64,
    raw_size: f64,
    prev_smoothed: Option<(f64, f64, f64)>,
    alpha: f64,
) -> (f64, f64, f64) {
    match prev_smoothed {
        None => (raw_cx, raw_cy, raw_size),
        Some((prev_cx, prev_cy, prev_size)) => (
            alpha * prev_cx + (1.0 - alpha) * raw_cx,
            alpha * prev_cy + (1.0 - alpha) * raw_cy,
            alpha * prev_size + (1.0 - alpha) * raw_size,
        ),
    }
}

// ---------------------------------------------------------------------------
// Hand bbox diagonal
// ---------------------------------------------------------------------------

/// Compute the bounding box diagonal of hand landmarks in pixel space.
/// Returns 0.0 if fewer than 2 valid (non-NaN) points.
pub fn hand_bbox_diagonal(landmarks_xyz: ArrayView2<f64>) -> f64 {
    let n = landmarks_xyz.nrows();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut count = 0usize;

    for i in 0..n {
        let x = landmarks_xyz[[i, 0]];
        let y = landmarks_xyz[[i, 1]];
        if !x.is_nan() && !y.is_nan() {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            count += 1;
        }
    }

    if count < 2 {
        return 0.0;
    }
    ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt()
}

// ---------------------------------------------------------------------------
// Hand crop size
// ---------------------------------------------------------------------------

/// Compute the hand crop size from multiple independent measures.
///
/// Each measure uses its own multiplier — they are not stacked.
/// The maximum of the three is used for robustness.
pub fn compute_hand_crop_size(
    arm_length: f64,
    last_hand_diagonal: f64,
    image_h: i32,
    hand_roi_scale: f64,
    hand_bbox_padding: f64,
    min_hand_crop_image_fraction: f64,
) -> f64 {
    let arm_crop = arm_length * hand_roi_scale;
    let hand_crop = if last_hand_diagonal > 0.0 {
        last_hand_diagonal * hand_bbox_padding
    } else {
        0.0
    };
    let min_crop = image_h as f64 * min_hand_crop_image_fraction;
    arm_crop.max(hand_crop).max(min_crop)
}

// ---------------------------------------------------------------------------
// Face crop helpers
// ---------------------------------------------------------------------------

/// Collect 2D positions of head landmarks exceeding the visibility threshold.
///
/// Returns `None` if fewer than 2 points are visible.
pub fn collect_visible_head_points(
    body_xyz: ArrayView2<f64>,
    body_vis: &[f64],
    head_indices: &[usize],
    visibility_threshold: f64,
) -> Option<Vec<[f64; 2]>> {
    let mut points: Vec<[f64; 2]> = Vec::new();
    for &idx in head_indices {
        if idx < body_xyz.nrows() && body_vis.get(idx).copied().unwrap_or(0.0) >= visibility_threshold {
            let x = body_xyz[[idx, 0]];
            let y = body_xyz[[idx, 1]];
            if !x.is_nan() && !y.is_nan() {
                points.push([x, y]);
            }
        }
    }
    if points.len() < 2 {
        None
    } else {
        Some(points)
    }
}

/// Compute the face ROI center and crop size from visible head landmarks.
///
/// Returns `((center_x, center_y), crop_size)` or `None` if < 2 points.
pub fn compute_face_crop_params(
    visible_head_points: &[[f64; 2]],
    face_roi_scale: f64,
) -> Option<((f64, f64), f64)> {
    if visible_head_points.len() < 2 {
        return None;
    }

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for &[x, y] in visible_head_points {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    let cx = (min_x + max_x) / 2.0;
    let cy = (min_y + max_y) / 2.0;
    let w = max_x - min_x;
    let h = max_y - min_y;
    let crop_size = w.max(h) * face_roi_scale;

    if crop_size <= 1.0 {
        return None;
    }

    Some(((cx, cy), crop_size))
}
