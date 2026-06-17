//! CompositeGPU session — three-model ONNX pipeline under a single CUDA context.
//!
//! Ported from `composite_gpu_session.py`. Architecture:
//!   body → RTMO one-stage → 17 COCO body keypoints
//!   hands → MediaPipe Hand ONNX → 21 keypoints per hand (ROI crop from wrists)
//!   face → RTMPose Face LaPa 106 → 106 keypoints (ROI crop from head landmarks)
//!
//! Phase 2: Single-image inference (N=1). Phase 3: batched inference.

use std::path::PathBuf;
use std::sync::Mutex;

use ndarray::{Array2, Array3, Array4};
use opencv::prelude::*;
use ort::session::Session;

use crate::onnx_utils::model_registry::{PreprocessMode, TrackerPreset, ModelSpec, resolve_model_path};
use crate::onnx_utils::session_builder::{Provider, build_tuned_ort_session};
use crate::onnx_utils::preprocessing::rtmo_preprocess;
use crate::onnx_utils::postprocessing::{rtmo_postprocess, get_simcc_maximum};
use crate::trackers::composite_gpu::roi::{
    RoiBox, compute_square_roi, smooth_roi_params,
    collect_visible_head_points, compute_face_crop_params,
};

// ---------------------------------------------------------------------------
// Session config
// ---------------------------------------------------------------------------

pub struct CompositeGpuSessionConfig {
    pub execution_provider: Provider,
    pub body_spec: ModelSpec,
    pub hand_spec: ModelSpec,
    pub face_spec: ModelSpec,
    pub detect_hands: bool,
    pub detect_face: bool,
    pub engine_cache_dir: PathBuf,
    pub fp16: bool,
    // ROI params
    pub roi_visibility_threshold: f64,
    pub roi_smoothing: f64,
    pub hand_roi_face_scale: f64,
    pub hand_roi_image_fraction: f64,
    pub face_roi_scale: f64,
    // Body keypoint names (resolved to indices at build time)
    pub body_left_wrist_name: String,
    pub body_right_wrist_name: String,
    pub body_left_elbow_name: String,
    pub body_right_elbow_name: String,
    pub body_head_point_names: Vec<String>,
    // Thresholds
    pub body_nms_thr: f32,
    pub body_score_thr: f32,
}

impl Default for CompositeGpuSessionConfig {
    fn default() -> Self {
        Self {
            execution_provider: Provider::CUDA,
            body_spec: ModelSpec::rtmo_medium(),
            hand_spec: ModelSpec::mediapipe_hand_landmark(),
            face_spec: ModelSpec::rtmpose_face(),
            detect_hands: true,
            detect_face: true,
            engine_cache_dir: dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("skellytracker")
                .join("trt_engines"),
            fp16: true,
            roi_visibility_threshold: 0.5,
            roi_smoothing: 0.5,
            hand_roi_face_scale: 2.0,
            hand_roi_image_fraction: 0.3,
            face_roi_scale: 2.5,
            body_left_wrist_name: "left_wrist".into(),
            body_right_wrist_name: "right_wrist".into(),
            body_left_elbow_name: "left_elbow".into(),
            body_right_elbow_name: "right_elbow".into(),
            body_head_point_names: vec![
                "nose".into(), "left_eye".into(), "right_eye".into(),
                "left_ear".into(), "right_ear".into(),
            ],
            body_nms_thr: 0.45,
            body_score_thr: 0.7,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-image detection results (before PointCloud assembly)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BodyResult {
    pub keypoints: Array3<f64>,   // (1, 17, 2) — first person only
    pub scores: Array2<f32>,      // (1, 17)
}

#[derive(Debug, Clone)]
pub struct HandResult {
    pub right_keypoints: Array2<f64>,  // (21, 2) — NaN if not detected
    pub left_keypoints: Array2<f64>,   // (21, 2)
    pub right_scores: Vec<f32>,        // (21,)
    pub left_scores: Vec<f32>,         // (21,)
    pub right_roi: Option<RoiBox>,
    pub left_roi: Option<RoiBox>,
}

#[derive(Debug, Clone)]
pub struct FaceResult {
    pub keypoints: Array2<f64>,  // (106, 2)
    pub scores: Array2<f32>,     // (1, 106)
    pub roi: Option<RoiBox>,
}

// ---------------------------------------------------------------------------
// Composite GPU Session
// ---------------------------------------------------------------------------

pub struct CompositeGpuSession {
    config: CompositeGpuSessionConfig,

    // ORT sessions (None if disabled)
    body_session: Option<Session>,
    hand_session: Option<Session>,
    face_session: Option<Session>,

    // Anatomical indices (resolved from body keypoint names at build time)
    body_left_wrist: usize,
    body_right_wrist: usize,
    body_left_elbow: usize,
    body_right_elbow: usize,
    body_head_indices: Vec<usize>,

    // Smoothed ROI state (frame-persistent EMA)
    smooth_face_roi: Mutex<Option<(f64, f64, f64)>>,
}

impl CompositeGpuSession {
    // =========================================================================
    // Construction
    // =========================================================================

    pub fn create(
        config: CompositeGpuSessionConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let provider = config.execution_provider;

        std::fs::create_dir_all(&config.engine_cache_dir)?;

        // Resolve model paths (sequential — downloads are I/O-bound)
        let body_path = resolve_model_path(body_key_for(&config.body_spec))?;
        let hand_path = if config.detect_hands {
            Some(resolve_model_path(hand_key_for(&config.hand_spec))?)
        } else {
            None
        };
        let face_path = if config.detect_face {
            Some(resolve_model_path(face_key_for(&config.face_spec))?)
        } else {
            None
        };

        eprintln!(
            "[skellytracker-rust] Building CompositeGPU session (provider={:?}, hands={}, face={})",
            provider, config.detect_hands, config.detect_face,
        );

        // Build ORT sessions
        let body_session = Some(build_tuned_ort_session(
            &body_path, provider, Some(&config.engine_cache_dir),
            config.fp16, "rtmo_body",
        )?);
        let hand_session = hand_path
            .map(|p| build_tuned_ort_session(&p, provider, Some(&config.engine_cache_dir), config.fp16, "hand"))
            .transpose()?;
        let face_session = face_path
            .map(|p| build_tuned_ort_session(&p, provider, Some(&config.engine_cache_dir), config.fp16, "face"))
            .transpose()?;

        // Resolve anatomical indices
        let (body_left_wrist, body_right_wrist, body_left_elbow, body_right_elbow, body_head_indices) =
            resolve_anatomical_indices(&config);

        eprintln!("[skellytracker-rust] CompositeGPU session ready");

        Ok(Self {
            config,
            body_session,
            hand_session,
            face_session,
            body_left_wrist,
            body_right_wrist,
            body_left_elbow,
            body_right_elbow,
            body_head_indices,
            smooth_face_roi: Mutex::new(None),
        })
    }

    pub fn preset(tier: TrackerPreset) -> CompositeGpuSessionConfig {
        let mut cfg = CompositeGpuSessionConfig::default();
        cfg.body_spec = ModelSpec::body_for_preset(tier);
        cfg
    }

    // =========================================================================
    // Single-image inference
    // =========================================================================

    /// Run full body + hands + face inference on a single BGR uint8 image.
    /// Returns (body, hands, face) results.
    pub fn predict(
        &mut self,
        image: &opencv::core::Mat,
    ) -> (BodyResult, HandResult, FaceResult) {
        // 1. Body
        let body = self.run_body(image);

        // 2. Hands + face from body landmarks
        let body_kpts_for_roi = if body.keypoints.shape()[0] > 0 {
            body.keypoints.slice(ndarray::s![0, .., ..]).to_owned()
        } else {
            Array2::from_elem((17, 2), f64::NAN)
        };
        let hands = self.run_hands(image, &body_kpts_for_roi);
        let face = self.run_face(image, &body_kpts_for_roi);

        (body, hands, face)
    }

    // =========================================================================
    // Body inference (single-image RTMO)
    // =========================================================================

    fn run_body(&mut self, image: &opencv::core::Mat) -> BodyResult {
        let session = match &mut self.body_session {
            Some(s) => s,
            None => return BodyResult {
                keypoints: Array3::from_elem((0, 17, 2), 0.0),
                scores: Array2::from_elem((0, 17), 0.0),
            },
        };

        let spec = &self.config.body_spec;
        let input_size = spec.input_size;

        // RTMO preprocess: letterbox + normalize
        let preprocess_result = rtmo_preprocess(
            image, input_size,
            spec.mean.as_ref(),
            spec.std.as_ref(),
        );
        let (padded, ratio) = match preprocess_result {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[skellytracker-rust] RTMO preprocess failed: {e}");
                return BodyResult {
                    keypoints: Array3::from_elem((0, 17, 2), 0.0),
                    scores: Array2::from_elem((0, 17), 0.0),
                };
            }
        };

        // HWC → CHW → (1, C, H, W) float32
        let h = padded.rows() as usize;
        let w = padded.cols() as usize;
        let step = padded.mat_step()[0] as usize;
        let mut flat = vec![0f32; 3 * h * w];
        unsafe {
            let ptr = padded.data() as *const f32;
            for c in 0..3 {
                let offset = c * h * w;
                for r in 0..h {
                    let src = std::slice::from_raw_parts(ptr.add(r * step / 4), w * 3);
                    let dst_row = &mut flat[offset + r * w..offset + (r + 1) * w];
                    for col in 0..w {
                        dst_row[col] = src[col * 3 + c];
                    }
                }
            }
        }

        let input_arr = Array4::from_shape_vec([1, 3, h, w], flat)
            .expect("rtmo tensor shape");
        let tensor = ort::value::Tensor::from_array(input_arr)
            .expect("rtmo tensor creation");

        let outputs = match session.run(ort::inputs![tensor]) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("[skellytracker-rust] RTMO inference failed: {e}");
                return BodyResult {
                    keypoints: Array3::from_elem((0, 17, 2), 0.0),
                    scores: Array2::from_elem((0, 17), 0.0),
                };
            }
        };

        // Extract det + pose outputs
        let det_view = outputs[0].try_extract_array::<f32>()
            .expect("rtmo det output");
        let pose_view = outputs[1].try_extract_array::<f32>()
            .expect("rtmo pose output");

        let det_shape = det_view.shape().to_vec();
        let pose_shape = pose_view.shape().to_vec();

        let det_array = Array3::from_shape_vec(
            (det_shape[0], det_shape[1], det_shape[2]),
            det_view.as_slice().unwrap_or(&[]).to_vec(),
        ).unwrap_or_else(|_| Array3::from_elem((1, 0, 5), 0.0));

        let pose_array = Array4::from_shape_vec(
            (pose_shape[0], pose_shape[1], pose_shape[2], pose_shape[3]),
            pose_view.as_slice().unwrap_or(&[]).to_vec(),
        ).unwrap_or_else(|_| Array4::from_elem((1, 0, 17, 3), 0.0));

        let (kpts_per_person, scores_per_person) = rtmo_postprocess(
            &det_array, &pose_array, ratio,
            self.config.body_nms_thr, self.config.body_score_thr,
        );

        if kpts_per_person.is_empty() {
            return BodyResult {
                keypoints: Array3::from_elem((0, 17, 2), 0.0),
                scores: Array2::from_elem((0, 17), 0.0),
            };
        }

        let n = kpts_per_person.len();
        let mut keypoints = Array3::<f64>::zeros((n, 17, 2));
        let mut scores = Array2::<f32>::zeros((n, 17));
        for i in 0..n {
            for k in 0..17 {
                keypoints[[i, k, 0]] = kpts_per_person[i][[k, 0]];
                keypoints[[i, k, 1]] = kpts_per_person[i][[k, 1]];
                scores[[i, k]] = scores_per_person[i][k];
            }
        }

        BodyResult { keypoints, scores }
    }

    // =========================================================================
    // Hand inference (single-image, MediaPipe ONNX)
    // =========================================================================

    fn run_hands(
        &mut self,
        image: &opencv::core::Mat,
        body_kpts: &Array2<f64>,
    ) -> HandResult {
        let spec = &self.config.hand_spec;
        let num_kpts = spec.num_keypoints as usize;
        let nan_kpts = Array2::from_elem((num_kpts, 2), f64::NAN);
        let zero_scores = vec![0.0f32; num_kpts];

        let session = match &mut self.hand_session {
            Some(s) => s,
            None => return HandResult {
                right_keypoints: nan_kpts.clone(),
                left_keypoints: nan_kpts,
                right_scores: zero_scores.clone(),
                left_scores: zero_scores,
                right_roi: None,
                left_roi: None,
            },
        };

        let image_h = image.rows();
        let image_w = image.cols();

        // Compute crop size from smoothed face ROI or image fraction
        let crop_sz = {
            let face_roi = self.smooth_face_roi.lock().unwrap();
            if let Some((_, _, sz)) = *face_roi {
                (sz * self.config.hand_roi_face_scale) as i32
            } else {
                (image_w.min(image_h) as f64 * self.config.hand_roi_image_fraction) as i32
            }
        };

        // Helper: detect one hand
        let mut detect_one = |wrist_idx: usize, elbow_idx: usize| -> (Array2<f64>, Vec<f32>, Option<RoiBox>) {
            if wrist_idx >= body_kpts.nrows() || elbow_idx >= body_kpts.nrows() {
                return (nan_kpts.clone(), zero_scores.clone(), None);
            }
            let wx = body_kpts[[wrist_idx, 0]];
            let wy = body_kpts[[wrist_idx, 1]];
            if wx.is_nan() || wy.is_nan() {
                return (nan_kpts.clone(), zero_scores.clone(), None);
            }

            let roi = compute_square_roi(
                wx as i32, wy as i32, crop_sz, image_w, image_h,
            );

            // Extract crop region as BGR
            let crop_rect = opencv::core::Rect::new(roi.x, roi.y, roi.width, roi.height);
            let cropped = match image.roi(crop_rect) {
                Ok(c) => c,
                Err(_) => return (nan_kpts.clone(), zero_scores.clone(), None),
            };

            // MediaPipe preprocessing: BGR→RGB, resize to (224,224), /255.0
            let model_h = spec.input_size.0 as i32;
            let model_w = spec.input_size.1 as i32;
            let mut rgb = opencv::core::Mat::default();
            if opencv::imgproc::cvt_color(&cropped, &mut rgb, opencv::imgproc::COLOR_BGR2RGB, 0, opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT).is_err() {
                return (nan_kpts.clone(), zero_scores.clone(), None);
            }
            let mut resized = opencv::core::Mat::default();
            if opencv::imgproc::resize(
                &rgb, &mut resized,
                opencv::core::Size::new(model_w, model_h),
                0.0, 0.0, opencv::imgproc::INTER_LINEAR,
            ).is_err() {
                return (nan_kpts.clone(), zero_scores.clone(), None);
            }

            // Convert to float32 [0,1] CHW
            let rh = resized.rows() as usize;
            let rw = resized.cols() as usize;
            let rstep = resized.mat_step()[0] as usize;
            let mut flat = vec![0f32; 3 * rh * rw];
            unsafe {
                let ptr = resized.data() as *const u8;
                for c in 0..3 {
                    let offset = c * rh * rw;
                    for row in 0..rh {
                        let src = std::slice::from_raw_parts(ptr.add(row * rstep), rw * 3);
                        let dst_row = &mut flat[offset + row * rw..offset + (row + 1) * rw];
                        for col in 0..rw {
                            dst_row[col] = src[col * 3 + c] as f32 / 255.0;
                        }
                    }
                }
            }

            let input_arr = match Array4::from_shape_vec([1, 3, rh, rw], flat) {
                Ok(a) => a,
                Err(_) => return (nan_kpts.clone(), zero_scores.clone(), None),
            };
            let tensor = match ort::value::Tensor::from_array(input_arr) {
                Ok(t) => t,
                Err(_) => return (nan_kpts.clone(), zero_scores.clone(), None),
            };

            let outputs = match session.run(ort::inputs![tensor]) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("[skellytracker-rust] Hand inference failed: {e}");
                    return (nan_kpts.clone(), zero_scores.clone(), None);
                }
            };

            // Decode MediaPipe hand: outputs[0] = xyz_x21 (1, 63) flattened [x,y,z,...]
            {
                if outputs.len() == 0 {
                    return (nan_kpts.clone(), zero_scores.clone(), None);
                }
                let xyz_view = match outputs[0].try_extract_array::<f32>() {
                    Ok(v) => v,
                    Err(_) => return (nan_kpts.clone(), zero_scores.clone(), None),
                };
                let slice = match xyz_view.as_slice() {
                    Some(s) => s,
                    None => return (nan_kpts.clone(), zero_scores.clone(), None),
                };
                let mh: f64 = 224.0;
                let mw: f64 = 224.0;
                let mut kpts = Array2::<f64>::zeros((num_kpts, 2));
                let mut sc = vec![0.0f32; num_kpts];
                for k in 0..num_kpts {
                    let x = slice[k * 3] as f64 / mw;
                    let y = slice[k * 3 + 1] as f64 / mh;
                    kpts[[k, 0]] = x * roi.width as f64 + roi.x as f64;
                    kpts[[k, 1]] = y * roi.height as f64 + roi.y as f64;
                    let z = slice[k * 3 + 2];
                    sc[k] = 1.0 / (1.0 + (-z).exp()); // sigmoid
                }
                (kpts, sc, Some(roi))
            }
        };

        let (right_kpts, right_sc, right_roi) = detect_one(self.body_right_wrist, self.body_right_elbow);
        let (left_kpts, left_sc, left_roi) = detect_one(self.body_left_wrist, self.body_left_elbow);

        HandResult {
            right_keypoints: right_kpts,
            left_keypoints: left_kpts,
            right_scores: right_sc,
            left_scores: left_sc,
            right_roi,
            left_roi,
        }
    }

    // =========================================================================
    // Face inference (single-image, RTMPose Face LaPa 106 SIMCC)
    // =========================================================================

    fn run_face(
        &mut self,
        image: &opencv::core::Mat,
        body_kpts: &Array2<f64>,
    ) -> FaceResult {
        let spec = &self.config.face_spec;
        let num_kpts = spec.num_keypoints as usize;
        let nan_kpts = Array2::from_elem((num_kpts, 2), f64::NAN);

        let session = match &mut self.face_session {
            Some(s) => s,
            None => return FaceResult {
                keypoints: nan_kpts,
                scores: Array2::from_elem((1, num_kpts), 0.0),
                roi: None,
            },
        };

        let image_h = image.rows();
        let image_w = image.cols();

        // Build visibility array from body keypoints (all visible for now)
        let body_vis: Vec<f64> = (0..body_kpts.nrows()).map(|i| {
            if body_kpts[[i, 0]].is_nan() { 0.0 } else { 1.0 }
        }).collect();

        let head_pts = match collect_visible_head_points(
            body_kpts.view(), &body_vis, &self.body_head_indices,
            self.config.roi_visibility_threshold,
        ) {
            Some(pts) => pts,
            None => return FaceResult {
                keypoints: nan_kpts,
                scores: Array2::from_elem((1, num_kpts), 0.0),
                roi: None,
            },
        };

        let ((raw_cx, raw_cy), raw_size) = match compute_face_crop_params(
            &head_pts, self.config.face_roi_scale,
        ) {
            Some(p) => p,
            None => return FaceResult {
                keypoints: nan_kpts,
                scores: Array2::from_elem((1, num_kpts), 0.0),
                roi: None,
            },
        };

        // Apply downward shift (20% of head width) — face extends below eyeline
        let head_w = head_pts.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max)
            - head_pts.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        let shifted_cy = raw_cy + head_w * 0.2;

        // Clamp crop size
        let crop_size = raw_size.clamp(120.0, 600.0);

        // EMA-smooth
        let (smooth_cx, smooth_cy, smooth_sz) = {
            let prev = *self.smooth_face_roi.lock().unwrap();
            smooth_roi_params(raw_cx, shifted_cy, crop_size, prev, self.config.roi_smoothing)
        };
        *self.smooth_face_roi.lock().unwrap() = Some((smooth_cx, smooth_cy, smooth_sz));

        let roi = compute_square_roi(
            smooth_cx as i32, smooth_cy as i32, smooth_sz as i32,
            image_w, image_h,
        );

        // Extract crop
        let crop_rect = opencv::core::Rect::new(roi.x, roi.y, roi.width, roi.height);
        let cropped = match image.roi(crop_rect) {
            Ok(c) => c,
            Err(_) => return FaceResult {
                keypoints: nan_kpts,
                scores: Array2::from_elem((1, num_kpts), 0.0),
                roi: Some(roi),
            },
        };

        // Simple letterbox for face
        let model_h = spec.input_size.0 as i32;
        let model_w = spec.input_size.1 as i32;
        let ratio = (model_h as f64 / cropped.rows() as f64)
            .min(model_w as f64 / cropped.cols() as f64);
        let nw = (cropped.cols() as f64 * ratio) as i32;
        let nh = (cropped.rows() as f64 * ratio) as i32;

        let mut resized = opencv::core::Mat::default();
        if opencv::imgproc::resize(
            &cropped, &mut resized,
            opencv::core::Size::new(nw, nh),
            0.0, 0.0, opencv::imgproc::INTER_LINEAR,
        ).is_err() {
            return FaceResult {
                keypoints: nan_kpts,
                scores: Array2::from_elem((1, num_kpts), 0.0),
                roi: Some(roi),
            };
        }

        // Pad to model size + normalize
        let mut padded = opencv::core::Mat::new_rows_cols_with_default(
            model_h, model_w, opencv::core::CV_32FC3,
            opencv::core::Scalar::new(114.0, 114.0, 114.0, 0.0),
        ).unwrap();
        let pad_roi = opencv::core::Rect::new(0, 0, nw, nh);
        let mut padded_roi = opencv::core::Mat::roi_mut(&mut padded, pad_roi).unwrap();
        if resized.convert_to(&mut padded_roi, opencv::core::CV_32F, 1.0, 0.0).is_err() {
            return FaceResult {
                keypoints: nan_kpts,
                scores: Array2::from_elem((1, num_kpts), 0.0),
                roi: Some(roi),
            };
        }

        // Normalize with BGR mean/std
        if let (Some(mean), Some(std)) = (spec.mean.as_ref(), spec.std.as_ref()) {
            let mean_s = opencv::core::Scalar::new(mean[0] as f64, mean[1] as f64, mean[2] as f64, 0.0);
            let inv_s = opencv::core::Scalar::new(
                1.0 / std[0] as f64, 1.0 / std[1] as f64, 1.0 / std[2] as f64, 0.0,
            );
            let mut temp = opencv::core::Mat::default();
            let _ = opencv::core::subtract(&padded, &mean_s, &mut temp, &opencv::core::Mat::default(), -1);
            let _ = opencv::core::multiply(&temp, &inv_s, &mut padded, 1.0, -1);
        }

        // HWC → CHW float32
        let ph = padded.rows() as usize;
        let pw = padded.cols() as usize;
        let pstep = padded.mat_step()[0] as usize;
        let mut flat = vec![0f32; 3 * ph * pw];
        unsafe {
            let ptr = padded.data() as *const f32;
            for c in 0..3 {
                let offset = c * ph * pw;
                for row in 0..ph {
                    let src = std::slice::from_raw_parts(ptr.add(row * pstep / 4), pw * 3);
                    let dst_row = &mut flat[offset + row * pw..offset + (row + 1) * pw];
                    for col in 0..pw {
                        dst_row[col] = src[col * 3 + c];
                    }
                }
            }
        }

        let input_arr = Array4::from_shape_vec([1, 3, ph, pw], flat)
            .expect("face tensor shape");
        let tensor = ort::value::Tensor::from_array(input_arr)
            .expect("face tensor creation");

        let outputs = match session.run(ort::inputs![tensor]) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("[skellytracker-rust] Face inference failed: {e}");
                return FaceResult {
                    keypoints: nan_kpts,
                    scores: Array2::from_elem((1, num_kpts), 0.0),
                    roi: Some(roi),
                };
            }
        };

        // SIMCC decode
        let simcc_x_view = outputs[0].try_extract_array::<f32>()
            .expect("simcc_x");
        let simcc_y_view = outputs[1].try_extract_array::<f32>()
            .expect("simcc_y");

        let sx_shape = simcc_x_view.shape().to_vec();
        let sy_shape = simcc_y_view.shape().to_vec();

        let simcc_x = Array3::from_shape_vec(
            (sx_shape[0], sx_shape[1], sx_shape[2]),
            simcc_x_view.as_slice().unwrap().to_vec(),
        ).unwrap();
        let simcc_y = Array3::from_shape_vec(
            (sy_shape[0], sy_shape[1], sy_shape[2]),
            simcc_y_view.as_slice().unwrap().to_vec(),
        ).unwrap();

        let (locs, scores) = get_simcc_maximum(&simcc_x, &simcc_y);
        // locs: (1, K, 2) in SIMCC label space

        let split_ratio = spec.simcc_split_ratio.unwrap_or(2.0) as f64;
        let mut face_kpts = Array2::<f64>::zeros((num_kpts, 2));
        let mut face_sc = Array2::<f32>::zeros((1, num_kpts));

        for k in 0..num_kpts {
            let x = locs[[0, k, 0]] as f64 / split_ratio / ratio;
            let y = locs[[0, k, 1]] as f64 / split_ratio / ratio;
            face_kpts[[k, 0]] = x + roi.x as f64;
            face_kpts[[k, 1]] = y + roi.y as f64;
            face_sc[[0, k]] = scores[[0, k]];
        }

        FaceResult {
            keypoints: face_kpts,
            scores: face_sc,
            roi: Some(roi),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn body_key_for(spec: &ModelSpec) -> &str {
    if spec.input_size == (640, 640) && spec.num_keypoints == 17 {
        "rtmo-m" // default
    } else {
        "rtmo-m"
    }
}

fn hand_key_for(spec: &ModelSpec) -> &str {
    match spec.preprocess_mode {
        PreprocessMode::MediaPipe => "mediapipe-hand-landmark",
        _ => "rtmpose-hand",
    }
}

fn face_key_for(_spec: &ModelSpec) -> &str {
    "rtmpose-face"
}

/// Resolve body keypoint name→index mappings from the RTMO body definition.
fn resolve_anatomical_indices(config: &CompositeGpuSessionConfig) -> (usize, usize, usize, usize, Vec<usize>) {
    // RTMO body 17 keypoint order: nose, left_eye, right_eye, left_ear, right_ear,
    //   left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,
    //   left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle
    const BODY_NAMES: &[&str] = &[
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ];

    let find = |name: &str| -> usize {
        BODY_NAMES.iter().position(|&n| n == name).unwrap_or(0)
    };

    let left_wrist = find(&config.body_left_wrist_name);
    let right_wrist = find(&config.body_right_wrist_name);
    let left_elbow = find(&config.body_left_elbow_name);
    let right_elbow = find(&config.body_right_elbow_name);
    let head_indices: Vec<usize> = config.body_head_point_names.iter()
        .map(|n| find(n))
        .collect();

    (left_wrist, right_wrist, left_elbow, right_elbow, head_indices)
}

