//! RTMPose tracker — two-stage ONNX pipeline (YOLOX detection + RTMPose keypoints).
//!
//! Phase 1: CPU single-image inference following the established BPT/Charuco pattern.

pub mod observation;

use ndarray::{Array2, Array3, Array4};
use opencv::core::{Mat, Scalar, Point};
use opencv::prelude::*;
use opencv::imgproc;
use ort::value::Tensor;

use crate::onnx_utils::{
    RtmPoseOrtSession, POSE_MEAN, POSE_STD, DET_NMS_THR, DET_SCORE_THR, SIMCC_SPLIT_RATIO,
};
use crate::onnx_utils::preprocessing::{yolox_letterbox_preprocess, rtmpose_letterbox_preprocess};
use crate::onnx_utils::postprocessing::{yolox_postprocess, rtmpose_letterbox_postprocess};
use crate::trackers::rtmpose::observation::RtmPoseObservation;
use crate::traits::{Observation, Tracker};

// ---------------------------------------------------------------------------
// Skeleton connections for drawing (from _skeleton_viz.py _COCO133_SKELETON_LINKS)
// These indices use rtmlib's NATIVE ordering (0-132).
// ---------------------------------------------------------------------------

fn skeleton_links() -> Vec<(usize, usize)> {
    let mut links = Vec::new();
    let body_edges = [
        (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (15, 17), (16, 20), (15, 18), (16, 21), (17, 19), (20, 22),
        (18, 19), (21, 22),
    ];
    links.extend(body_edges.iter());
    for i in 23..90 { links.push((i, i + 1)); }
    for i in 91..111 { links.push((i, i + 1)); }
    for i in 112..132 { links.push((i, i + 1)); }
    links
}

// ---------------------------------------------------------------------------
// Drawing constants
// ---------------------------------------------------------------------------

const KEYPOINT_RADIUS: i32 = 3;
const KEYPOINT_THICKNESS: i32 = -1;
const SKELETON_COLOR: Scalar = Scalar::new(0.0, 255.0, 0.0, 0.0);
const SKELETON_THICKNESS: i32 = 2;
const KEYPOINT_COLOR: Scalar = Scalar::new(0.0, 0.0, 255.0, 0.0);

// ---------------------------------------------------------------------------
// RtmPoseTracker
// ---------------------------------------------------------------------------

pub struct RtmPoseTracker {
    pub mode: String,
    session: RtmPoseOrtSession,
    det_input_size: (u32, u32),
    pose_input_size: (u32, u32),
}

impl RtmPoseTracker {
    pub fn new(mode: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = RtmPoseOrtSession::new(mode)?;
        Ok(Self {
            mode: mode.to_string(),
            det_input_size: session.det_input_size,
            pose_input_size: session.pose_input_size,
            session,
        })
    }

    /// Run the full two-stage detection pipeline on a BGR image.
    pub fn detect(&mut self, frame_number: u64, image: &Mat) -> RtmPoseObservation {
        let image_size = (image.rows() as u32, image.cols() as u32);

        let bboxes = match self.run_yolox(image) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[skellytracker-rust] YOLOX detection failed: {e}");
                return RtmPoseObservation::empty(frame_number, image_size);
            }
        };

        if bboxes.is_empty() {
            return RtmPoseObservation::empty(frame_number, image_size);
        }

        let result = match self.run_rtmpose_single(image, &bboxes[0]) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[skellytracker-rust] RTMPose inference failed: {e}");
                return RtmPoseObservation::empty(frame_number, image_size);
            }
        };

        let (keypoints, scores) = result;
        RtmPoseObservation::from_detection_results(frame_number, keypoints, scores, image_size)
    }

    /// YOLOX detection → person bboxes in image coordinates.
    fn run_yolox(&mut self, image: &Mat) -> Result<Vec<[f64; 4]>, Box<dyn std::error::Error>> {
        let (padded, ratio) = yolox_letterbox_preprocess(image, self.det_input_size)?;

        // Mat → ndarray: extract raw pixels, HWC → CHW → add batch dim
        let h = padded.rows() as usize;
        let w = padded.cols() as usize;
        let step = padded.mat_step()[0] as usize;
        let mut data = vec![0u8; h * w * 3];
        unsafe {
            let ptr = padded.data() as *const u8;
            for r in 0..h {
                let src = std::slice::from_raw_parts(ptr.add(r * step), w * 3);
                let dst_start = r * w * 3;
                data[dst_start..dst_start + w * 3].copy_from_slice(src);
            }
        }

        let array_hwc = Array3::from_shape_vec((h, w, 3), data)?;
        let array_chw = array_hwc.permuted_axes([2, 0, 1]);
        let array_float = array_chw.mapv(|v| v as f32);

        // Build (1, 3, H, W) tensor
        let shape = [1, array_float.shape()[0], array_float.shape()[1], array_float.shape()[2]];
        let flat: Vec<f32> = array_float.iter().cloned().collect();
        let input_arr = Array4::from_shape_vec(shape, flat)?;

        let det_input = Tensor::from_array(input_arr)
            .map_err(|e| format!("YOLOX tensor from_array: {e}"))?;
        let outputs = self.session.det_session.run(ort::inputs![det_input])?;

        let det_view = outputs[0].try_extract_array::<f32>()?;
        let shape = det_view.shape();
        let n_dets = shape[1];

        let det_slice = det_view.as_slice().ok_or("non-contiguous det output")?;
        let det_array = Array3::from_shape_vec((1, n_dets, shape[2]), det_slice.to_vec())?;

        Ok(yolox_postprocess(&det_array, ratio, DET_NMS_THR, DET_SCORE_THR))
    }

    /// RTMPose keypoint estimation for a single person bbox.
    fn run_rtmpose_single(
        &mut self,
        image: &Mat,
        bbox: &[f64; 4],
    ) -> Result<(Array3<f64>, Array2<f32>), Box<dyn std::error::Error>> {
        let (cropped, center, scale) = rtmpose_letterbox_preprocess(image, bbox, self.pose_input_size)?;

        // Extract float32 pixels → ndarray → normalize → CHW → batch dim
        let h = cropped.rows() as usize;
        let w = cropped.cols() as usize;
        let step = cropped.mat_step();
        let step = step[0] as usize;

        let mut data = vec![0f32; h * w * 3];
        unsafe {
            let ptr = cropped.data() as *const u8;
            for r in 0..h {
                let row_bytes = std::slice::from_raw_parts(ptr.add(r * step), w * 3 * 4);
                let dst_start = r * w * 3;
                // reinterpret u8 bytes as f32
                let f32_ptr = row_bytes.as_ptr() as *const f32;
                let f32_row = std::slice::from_raw_parts(f32_ptr, w * 3);
                data[dst_start..dst_start + w * 3].copy_from_slice(f32_row);
            }
        }

        let mut array_hwc = Array3::from_shape_vec((h, w, 3), data)?;

        // Normalize: (pixel - mean) / std — done in ndarray space
        for c in 0..3 {
            let mean = POSE_MEAN[c] as f32;
            let std = POSE_STD[c] as f32;
            let inv_std = 1.0 / std;
            array_hwc.index_axis_mut(ndarray::Axis(2), c).mapv_inplace(|v| (v - mean) * inv_std);
        }

        let array_chw = array_hwc.permuted_axes([2, 0, 1]);
        let shape = [1, array_chw.shape()[0], array_chw.shape()[1], array_chw.shape()[2]];
        let flat: Vec<f32> = array_chw.iter().cloned().collect();
        let input_arr = Array4::from_shape_vec(shape, flat)?;

        let pose_input = Tensor::from_array(input_arr)
            .map_err(|e| format!("pose tensor from_array: {e}"))?;
        let outputs = self.session.pose_session.run(ort::inputs![pose_input])?;

        // Extract SIMCC outputs
        let simcc_x_view = outputs[0].try_extract_array::<f32>()?;
        let simcc_y_view = outputs[1].try_extract_array::<f32>()?;

        let sx = simcc_x_view.as_slice().ok_or("simcc_x not contiguous")?;
        let sy = simcc_y_view.as_slice().ok_or("simcc_y not contiguous")?;

        let sx_shape = simcc_x_view.shape().to_vec();
        let sy_shape = simcc_y_view.shape().to_vec();

        let simcc_x = Array3::from_shape_vec(
            (sx_shape[0], sx_shape[1], sx_shape[2]),
            sx.to_vec(),
        )?;
        let simcc_y = Array3::from_shape_vec(
            (sy_shape[0], sy_shape[1], sy_shape[2]),
            sy.to_vec(),
        )?;

        Ok(rtmpose_letterbox_postprocess(
            &simcc_x, &simcc_y, &center, &scale,
            self.pose_input_size, SIMCC_SPLIT_RATIO,
        ))
    }

    pub fn draw_markers_into(&self, image: &mut Mat, obs: &dyn Observation) {
        let o = match obs.as_any().downcast_ref::<RtmPoseObservation>() {
            Some(c) => c,
            None => return,
        };

        if o.keypoints.shape()[0] == 0 {
            return;
        }

        let links = skeleton_links();

        for &(i0, i1) in &links {
            if i0 >= 133 || i1 >= 133 { continue; }
            let x0 = o.keypoints[[0, i0, 0]] as f32;
            let y0 = o.keypoints[[0, i0, 1]] as f32;
            let x1 = o.keypoints[[0, i1, 0]] as f32;
            let y1 = o.keypoints[[0, i1, 1]] as f32;
            if x0.is_nan() || y0.is_nan() || x1.is_nan() || y1.is_nan() { continue; }

            let p0 = Point::new(x0 as i32, y0 as i32);
            let p1 = Point::new(x1 as i32, y1 as i32);
            if imgproc::line(image, p0, p1, SKELETON_COLOR, SKELETON_THICKNESS, imgproc::LINE_8, 0).is_err() {
                return;
            }
        }

        for k in 0..133 {
            let kx = o.keypoints[[0, k, 0]] as f32;
            let ky = o.keypoints[[0, k, 1]] as f32;
            if kx.is_nan() || ky.is_nan() { continue; }
            let center = Point::new(kx as i32, ky as i32);
            let _ = imgproc::circle(image, center, KEYPOINT_RADIUS, KEYPOINT_COLOR, KEYPOINT_THICKNESS, imgproc::LINE_8, 0);
        }
    }
}

impl Tracker for RtmPoseTracker {
    fn process_image(&mut self, frame_number: u64, image: &Mat) -> Box<dyn Observation> {
        Box::new(self.detect(frame_number, image))
    }

    fn annotate_image(&self, image: &Mat, obs: &dyn Observation) -> Mat {
        let mut annotated = image.clone();
        self.draw_markers_into(&mut annotated, obs);
        annotated
    }
}
