//! Preprocessing functions ported from Python's rtm_preprocessing.py.

use ndarray::Array2;
use opencv::core::{Mat, Size, Scalar, BORDER_DEFAULT, MatTraitConst};
use opencv::imgproc;

/// YOLOX letterbox: resize + pad with gray (114), no normalisation.
/// Returns `(padded_img, ratio)` where padded_img is BGR uint8.
pub fn yolox_letterbox_preprocess(
    img: &Mat,
    model_input_size: (u32, u32),
) -> Result<(Mat, f64), opencv::Error> {
    let (th, tw) = model_input_size;
    let img_h = img.rows();
    let img_w = img.cols();

    let ratio = f64::min(th as f64 / img_h as f64, tw as f64 / img_w as f64);
    let nw = (img_w as f64 * ratio) as i32;
    let nh = (img_h as f64 * ratio) as i32;

    let mut resized = Mat::default();
    imgproc::resize(img, &mut resized, Size::new(nw, nh), 0.0, 0.0, imgproc::INTER_LINEAR)?;

    let padded = Mat::new_rows_cols_with_default(th as i32, tw as i32, resized.typ(), Scalar::all(114.0))?;

    // Copy resized image into top-left of padded using raw pointers
    let src_step = resized.mat_step()[0] as usize;
    let dst_step = padded.mat_step()[0] as usize;
    for r in 0..nh as usize {
        unsafe {
            let src_ptr = (resized.data() as *const u8).add(r * src_step);
            let dst_ptr = (padded.data() as *mut u8).add(r * dst_step);
            let src_row = std::slice::from_raw_parts(src_ptr, nw as usize * 3);
            let dst_row = std::slice::from_raw_parts_mut(dst_ptr, nw as usize * 3);
            dst_row.copy_from_slice(src_row);
        }
    }

    Ok((padded, ratio))
}

/// Convert xyxy bbox → (center, scale) for top-down affine warp.
pub fn bbox_xyxy2cs(bbox: &[f64], padding: f64) -> (Array2<f64>, Array2<f64>) {
    let (x1, y1, x2, y2) = (bbox[0], bbox[1], bbox[2], bbox[3]);
    let center = Array2::from_shape_vec((1, 2), vec![(x1 + x2) * 0.5, (y1 + y2) * 0.5]).unwrap();
    let scale = Array2::from_shape_vec((1, 2), vec![(x2 - x1) * padding, (y2 - y1) * padding]).unwrap();
    (center, scale)
}

/// Affine-crop a person bbox from img, resize to input_size.
pub fn top_down_affine(
    input_size: (u32, u32),
    bbox_scale: &Array2<f64>,
    bbox_center: &Array2<f64>,
    img: &Mat,
) -> Result<(Mat, Array2<f64>), opencv::Error> {
    let (w, h) = input_size;
    let aspect_ratio = w as f64 / h as f64;
    let bw = bbox_scale[[0, 0]];
    let bh = bbox_scale[[0, 1]];

    let (new_bw, new_bh) = if bw > bh * aspect_ratio {
        (bw, bw / aspect_ratio)
    } else {
        (bh * aspect_ratio, bh)
    };

    let corrected_scale = Array2::from_shape_vec((1, 2), vec![new_bw, new_bh]).unwrap();

    let cx = bbox_center[[0, 0]];
    let cy = bbox_center[[0, 1]];

    // Build affine transform matrix matching Python's get_warp_matrix with rot=0.
    // Python _get_3rd_point(a, b): direction = a - b, c = b + [-direction[1], direction[0]]
    //
    // For src (a=[cx,cy], b=[cx, cy-sw/2]):
    //   direction = [0, sw/2], 3rd = [cx-sw/2, cy-sw/2]
    // For dst (a=[w/2,h/2], b=[w/2, h/2-w/2]):
    //   direction = [0, w/2], 3rd = [0, h/2-w/2]
    let sw2 = 0.5 * new_bw;
    let dw2 = 0.5 * w as f64;

    let src_pts = Mat::from_slice_2d(&[
        [cx as f32, cy as f32],                         // src[0] = center
        [cx as f32, (cy - sw2) as f32],                  // src[1] = center + [0, -sw/2]
        [(cx - sw2) as f32, (cy - sw2) as f32],          // src[2] = b + [-direction[1], direction[0]]
    ])?;

    let dst_pts = Mat::from_slice_2d(&[
        [(w as f32 * 0.5), (h as f32 * 0.5)],                          // dst[0] = [w/2, h/2]
        [(w as f32 * 0.5), (h as f32 * 0.5 - dw2 as f32)],             // dst[1] = [w/2, h/2] + [0, -w/2]
        [(0.0_f32), (h as f32 * 0.5 - dw2 as f32)],                    // dst[2] = [0, h/2 - w/2]
    ])?;

    let warp_mat = imgproc::get_affine_transform(&src_pts, &dst_pts)?;

    let warp_size = Size::new(w as i32, h as i32);
    let mut img_out = Mat::default();
    imgproc::warp_affine(
        img, &mut img_out, &warp_mat, warp_size,
        imgproc::INTER_LINEAR, BORDER_DEFAULT, Scalar::default(),
    )?;

    Ok((img_out, corrected_scale))
}

/// RTMPose top-down preprocessing: affine crop around bbox.
/// Returns the cropped (but NOT normalized) float32 Mat, plus center/scale for postprocessing.
/// Normalization is done later via ndarray when building the ONNX input tensor.
pub fn rtmpose_letterbox_preprocess(
    img: &Mat,
    bbox: &[f64],
    model_input_size: (u32, u32),
) -> Result<(Mat, Array2<f64>, Array2<f64>), opencv::Error> {
    let (center, scale) = bbox_xyxy2cs(bbox, 1.25);
    let (cropped, corrected_scale) = top_down_affine(model_input_size, &scale, &center, img)?;

    let mut float_img = Mat::default();
    cropped.convert_to(&mut float_img, opencv::core::CV_32F, 1.0, 0.0)?;

    Ok((float_img, center, corrected_scale))
}

// ---------------------------------------------------------------------------
// RTMO one-stage body preprocessing
// ---------------------------------------------------------------------------

/// Port of Python's `rtmo_preprocess()` — letterbox + normalize for RTMO.
///
/// Returns (padded_float32_img, ratio) where padded_img is (th, tw) float32
/// with gray=114 padding and optional BGR mean/std normalization.
pub fn rtmo_preprocess(
    img: &Mat,
    model_input_size: (u32, u32),
    mean: Option<&[f32; 3]>,
    std: Option<&[f32; 3]>,
) -> Result<(Mat, f64), opencv::Error> {
    let (th, tw) = (model_input_size.0 as i32, model_input_size.1 as i32);
    let img_h = img.rows();
    let img_w = img.cols();

    // Letterbox: compute ratio, resize, pad
    let ratio = (th as f64 / img_h as f64).min(tw as f64 / img_w as f64);
    let nw = (img_w as f64 * ratio) as i32;
    let nh = (img_h as f64 * ratio) as i32;

    let mut resized = Mat::default();
    imgproc::resize(img, &mut resized, Size::new(nw, nh), 0.0, 0.0, imgproc::INTER_LINEAR)?;

    // Create padded float32 image with gray=114 fill
    let mut padded = Mat::new_rows_cols_with_default(th, tw, opencv::core::CV_32FC3, Scalar::all(114.0))?;

    // Copy resized region into top-left of padded
    let roi = opencv::core::Rect::new(0, 0, nw, nh);
    let mut padded_roi = Mat::roi_mut(&mut padded, roi)?;
    resized.convert_to(&mut padded_roi, opencv::core::CV_32F, 1.0, 0.0)?;

    // Normalize with BGR mean/std
    if let (Some(m), Some(s)) = (mean, std) {
        let mean_scalar = Scalar::new(m[0] as f64, m[1] as f64, m[2] as f64, 0.0);
        let inv_std = Scalar::new(1.0 / s[0] as f64, 1.0 / s[1] as f64, 1.0 / s[2] as f64, 0.0);
        let mut temp = Mat::default();
        opencv::core::subtract(&padded, &mean_scalar, &mut temp, &Mat::default(), -1)?;
        opencv::core::multiply(&temp, &inv_std, &mut padded, 1.0, -1)?;
    }

    Ok((padded, ratio))
}
