//! Postprocessing functions ported from Python's rtm_postprocessing.py.

use ndarray::{Array1, Array2, Array3, Axis};

/// Decode SIMCC heatmaps to (x, y) coordinates with confidence scores.
/// Port of Python's `get_simcc_maximum`.
///
/// simcc_x: shape (N, K, Wx) — x-axis SIMCC heatmaps
/// simcc_y: shape (N, K, Wy) — y-axis SIMCC heatmaps
///
/// Returns (locs, vals) where:
///   locs: shape (N, K, 2) — x/y keypoint coordinates (in SIMCC label space)
///   vals: shape (N, K)   — confidence scores
pub fn get_simcc_maximum(
    simcc_x: &Array3<f32>,
    simcc_y: &Array3<f32>,
) -> (Array3<f32>, Array2<f32>) {
    let n = simcc_x.shape()[0];
    let k = simcc_x.shape()[1];
    let wx = simcc_x.shape()[2];
    let wy = simcc_y.shape()[2];

    // Reshape to (N*K, W)
    let simcc_x_2d = simcc_x
        .view()
        .into_shape((n * k, wx))
        .unwrap()
        .to_owned();
    let simcc_y_2d = simcc_y
        .view()
        .into_shape((n * k, wy))
        .unwrap()
        .to_owned();

    // argmax along axis=1
    let x_locs: Vec<i32> = simcc_x_2d
        .axis_iter(Axis(0))
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as i32)
                .unwrap_or(-1)
        })
        .collect();

    let y_locs: Vec<i32> = simcc_y_2d
        .axis_iter(Axis(0))
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as i32)
                .unwrap_or(-1)
        })
        .collect();

    // Max values
    let max_val_x: Vec<f32> = simcc_x_2d
        .axis_iter(Axis(0))
        .map(|col| col.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
        .collect();
    let max_val_y: Vec<f32> = simcc_y_2d
        .axis_iter(Axis(0))
        .map(|col| col.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
        .collect();

    let total = (n * k) as usize;
    let mut locs_data = Vec::with_capacity(total * 2);
    let mut vals_data = Vec::with_capacity(total);

    for i in 0..total {
        let val = 0.5 * (max_val_x[i] + max_val_y[i]);
        if val <= 0.0 {
            locs_data.push(-1.0_f32);
            locs_data.push(-1.0_f32);
        } else {
            locs_data.push(x_locs[i] as f32);
            locs_data.push(y_locs[i] as f32);
        }
        vals_data.push(val);
    }

    let locs = Array3::from_shape_vec((n as usize, k as usize, 2), locs_data).unwrap();
    let vals = Array2::from_shape_vec((n as usize, k as usize), vals_data).unwrap();

    (locs, vals)
}

/// Decode SIMCC outputs back to original image coordinates.
/// Port of Python's `rtmpose_letterbox_postprocess`.
///
/// Returns (keypoints, scores) where:
///   keypoints: (1, K, 2) float64 in image coordinates
///   scores: (1, K) float32
pub fn rtmpose_letterbox_postprocess(
    simcc_x: &Array3<f32>,
    simcc_y: &Array3<f32>,
    center: &Array2<f64>,   // (1, 2)
    scale: &Array2<f64>,    // (1, 2)
    model_input_size: (u32, u32),
    simcc_split_ratio: f32,
) -> (Array3<f64>, Array2<f32>) {
    let (locs, scores) = get_simcc_maximum(simcc_x, simcc_y);

    let n = locs.shape()[0];
    let k = locs.shape()[1];

    let cx = center[[0, 0]];
    let cy = center[[0, 1]];
    let s0 = scale[[0, 0]];
    let s1 = scale[[0, 1]];
    let mip_h = model_input_size.0 as f64;
    let mip_w = model_input_size.1 as f64;

    let mut keypoints = Array3::<f64>::zeros((n, k, 2));

    for i in 0..n {
        for j in 0..k {
            let lx = locs[[i, j, 0]] as f64;
            let ly = locs[[i, j, 1]] as f64;

            let kpx = lx / simcc_split_ratio as f64 / mip_w * s0 + cx - s0 / 2.0;
            let kpy = ly / simcc_split_ratio as f64 / mip_h * s1 + cy - s1 / 2.0;

            keypoints[[i, j, 0]] = kpx;
            keypoints[[i, j, 1]] = kpy;
        }
    }

    (keypoints, scores)
}

/// Single-class NMS implemented in Rust.
/// Port of Python's `nms()`.
pub fn nms(boxes: &Array2<f32>, scores: &Array1<f32>, nms_thr: f32) -> Vec<usize> {
    let n = boxes.shape()[0];
    if n == 0 {
        return Vec::new();
    }

    let x1 = boxes.column(0).to_owned();
    let y1 = boxes.column(1).to_owned();
    let x2 = boxes.column(2).to_owned();
    let y2 = boxes.column(3).to_owned();

    let areas: Vec<f32> = (0..n)
        .map(|i| (x2[i] - x1[i] + 1.0) * (y2[i] - y1[i] + 1.0))
        .collect();

    // Sort indices by score descending
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

    let mut keep: Vec<usize> = Vec::new();
    let mut suppressed = vec![false; n];

    for &i in &order {
        if suppressed[i] {
            continue;
        }
        keep.push(i);

        for &j in &order {
            if j == i || suppressed[j] {
                continue;
            }

            let xx1 = x1[i].max(x1[j]);
            let yy1 = y1[i].max(y1[j]);
            let xx2 = x2[i].min(x2[j]);
            let yy2 = y2[i].min(y2[j]);

            let w = (xx2 - xx1 + 1.0).max(0.0);
            let h = (yy2 - yy1 + 1.0).max(0.0);
            let inter = w * h;
            let ovr = inter / (areas[i] + areas[j] - inter);

            if ovr > nms_thr {
                suppressed[j] = true;
            }
        }
    }

    keep
}

/// YOLOX postprocessing for a single image.
/// Takes the raw YOLOX output and returns detected person bboxes in image coordinates.
///
/// det_output: shape (1, N, 5) — [x1, y1, x2, y2, score] already in model coordinates
/// ratio: letterbox scale factor
/// Returns Vec of [x1, y1, x2, y2] bboxes in image coordinates.
pub fn yolox_postprocess(
    det_output: &Array3<f32>,
    ratio: f64,
    nms_thr: f32,
    score_thr: f32,
) -> Vec<[f64; 4]> {
    if det_output.is_empty() {
        return Vec::new();
    }

    let n_dets = det_output.shape()[1]; // number of detections
    let mut boxes = Array2::<f32>::zeros((n_dets, 4));
    let mut scores = Array1::<f32>::zeros(n_dets);

    for i in 0..n_dets {
        boxes[[i, 0]] = det_output[[0, i, 0]] / ratio as f32;
        boxes[[i, 1]] = det_output[[0, i, 1]] / ratio as f32;
        boxes[[i, 2]] = det_output[[0, i, 2]] / ratio as f32;
        boxes[[i, 3]] = det_output[[0, i, 3]] / ratio as f32;
        scores[i] = det_output[[0, i, 4]];
    }

    let keep = nms(&boxes, &scores, nms_thr);

    keep.iter()
        .filter(|&&i| scores[i] > score_thr)
        .map(|&i| {
            [
                boxes[[i, 0]] as f64,
                boxes[[i, 1]] as f64,
                boxes[[i, 2]] as f64,
                boxes[[i, 3]] as f64,
            ]
        })
        .collect()
}
