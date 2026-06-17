pub mod observation;

use std::collections::HashSet;

use observation::CharucoObservation;

use opencv::core::{Mat, Point2f, Scalar, Size, Vector};
use opencv::imgproc;
use opencv::objdetect;
use opencv::prelude::*;

use crate::point_cloud::PointCloud;
use crate::traits::{Observation, Tracker};

// ── Drawing constants — match Python CharucoAnnotatorConfig ─────────────────

const CORNER_MARKER_COLOR: Scalar = Scalar::new(255.0, 0.0, 255.0, 0.0); // magenta
const CORNER_MARKER_SIZE: i32 = 10;
const CORNER_MARKER_THICKNESS: i32 = 2;

const ARUCO_LINES_COLOR: Scalar = Scalar::new(0.0, 255.0, 0.0, 0.0); // green
const ARUCO_LINES_THICKNESS: i32 = 2;

const TEXT_COLOR: Scalar = Scalar::new(40.0, 115.0, 215.0, 0.0); // BGR
const TEXT_FONT: i32 = imgproc::FONT_HERSHEY_SIMPLEX;
const TEXT_SCALE: f64 = 0.5;
const TEXT_THICKNESS: i32 = 2;

// ── Tracker ─────────────────────────────────────────────────────────────────

pub struct CharucoTracker {
    pub squares_x: u32,
    pub squares_y: u32,
    #[allow(dead_code)]
    square_length_mm: f32,
    #[allow(dead_code)]
    marker_length_ratio: f32,
    #[allow(dead_code)]
    dictionary_enum: i32,

    #[allow(dead_code)]
    board: objdetect::CharucoBoard,
    detector: objdetect::CharucoDetector,
    pub all_charuco_ids: Vec<i32>,
    pub all_aruco_ids: Vec<i32>,
    /// O(1) lookup for board membership test in marker loop
    all_aruco_set: HashSet<i32>,
    /// Cached corner names ("CharucoCorner-0", ...)
    corner_names: Vec<String>,

    /// Pre-computed from board.get_chessboard_corners().
    all_charuco_corners_3d: Vec<[f64; 3]>,
    /// Pre-computed from board.get_obj_points().
    all_aruco_corners_3d: Vec<[f64; 3]>,
}

impl CharucoTracker {
    pub fn new(
        squares_x: u32,
        squares_y: u32,
        square_length_mm: f32,
        marker_length_ratio: f32,
        dictionary_enum: i32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if squares_x < 2 || squares_y < 2 {
            return Err("Board must have at least 2x2 squares".into());
        }
        if marker_length_ratio <= 0.0 || marker_length_ratio >= 1.0 {
            return Err("marker_length_ratio must be between 0 and 1".into());
        }

        let dictionary = objdetect::get_predefined_dictionary_i32(dictionary_enum)?;
        let marker_length = marker_length_ratio * square_length_mm;

        let size = Size::new(squares_x as i32, squares_y as i32);
        let board = objdetect::CharucoBoard::new_def(size, square_length_mm, marker_length, &dictionary)?;
        let detector = objdetect::CharucoDetector::new_def(&board)?;

        // Collect board IDs
        let board_ids = board.get_ids()?;
        let all_aruco_ids: Vec<i32> = board_ids.iter().collect();

        let n_corners = ((squares_x - 1) * (squares_y - 1)) as usize;
        let all_charuco_ids: Vec<i32> = (0..n_corners as i32).collect();

        // Collect 3D object coordinates for the observation parity requirement
        let chessboard_corners_3d = board.get_chessboard_corners()?;
        let all_charuco_corners_3d: Vec<[f64; 3]> = (0..chessboard_corners_3d.len())
            .filter_map(|i| {
                chessboard_corners_3d.get(i).ok().map(|pt| {
                    [pt.x as f64, pt.y as f64, pt.z as f64]
                })
            })
            .collect();

        let obj_points = board.get_obj_points()?;
        let mut all_aruco_corners_3d: Vec<[f64; 3]> = Vec::with_capacity(obj_points.len());
        for i in 0..obj_points.len() {
            if let Ok(inner) = obj_points.get(i) {
                for j in 0..inner.len() {
                    if let Ok(pt) = inner.get(j) {
                        all_aruco_corners_3d.push([pt.x as f64, pt.y as f64, pt.z as f64]);
                    }
                }
            }
        }

        let all_aruco_set: HashSet<i32> = all_aruco_ids.iter().copied().collect();
        let corner_names: Vec<String> = format_corner_names(n_corners);

        Ok(CharucoTracker {
            squares_x,
            squares_y,
            square_length_mm,
            marker_length_ratio,
            dictionary_enum,
            board,
            detector,
            all_charuco_ids,
            all_aruco_ids,
            all_aruco_set,
            corner_names,
            all_charuco_corners_3d,
            all_aruco_corners_3d,
        })
    }

    /// Run charuco board detection — precisely replicates Python:
    ///
    /// ```python
    /// grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    /// (charuco_corners, charuco_ids, aruco_corners, aruco_ids) = detector.detectBoard(grey)
    /// ```
    pub fn detect(&self, frame_number: u64, image: &Mat) -> CharucoObservation {
        let _t0 = std::time::Instant::now();
        let image_h = image.rows() as u32;
        let image_w = image.cols() as u32;

        // 1. BGR → GRAY
        let mut gray = Mat::default();
        if imgproc::cvt_color(
            image, &mut gray,
            imgproc::COLOR_BGR2GRAY, 0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        ).is_err() {
            eprintln!("[skellytracker-rust] charuco cvt_color failed");
            return self.empty_obs(frame_number, image_h, image_w);
        }
        let _t1 = std::time::Instant::now();

        // ── Step 2: Detect markers + charuco corners in a single C++ call ──
        // Matches Python's single `detector.detectBoard(grey)` call.
        // detect_board runs internal marker detection + charuco interpolation
        // in one pass — no double-detection, no per-frame ArucoDetector allocs.
        let mut marker_corners: Vector<Vector<Point2f>> = Vector::new();
        let mut marker_ids_mat = Mat::default();
        let mut charuco_corners: Vector<Point2f> = Vector::new();
        let mut charuco_ids: Vector<i32> = Vector::new();

        if self.detector.detect_board(
            &gray,
            &mut charuco_corners,
            &mut charuco_ids,
            &mut marker_corners,
            &mut marker_ids_mat,
        ).is_err() {
            eprintln!("[skellytracker-rust] charuco detect_board failed");
            return self.empty_obs(frame_number, image_h, image_w);
        }
        let _t2 = std::time::Instant::now();

        // 3. Extract charuco corner positions and IDs from output vectors.
        let mut raw_corners: Vec<[f64; 2]> = Vec::with_capacity(charuco_corners.len());
        let mut detected_corners_img: Vec<[f64; 2]> = Vec::with_capacity(charuco_corners.len());
        for i in 0..charuco_corners.len() {
            if let Ok(pt) = charuco_corners.get(i) {
                raw_corners.push([pt.x as f64, pt.y as f64]);
                detected_corners_img.push([pt.x as f64, pt.y as f64]);
            }
        }

        let mut detected_ids: Vec<i32> = Vec::with_capacity(charuco_ids.len());
        for i in 0..charuco_ids.len() {
            if let Ok(id) = charuco_ids.get(i) {
                detected_ids.push(id);
            }
        }

        // 5. Compute detected_charuco_corners_in_object_coordinates
        //    = subset of all_charuco_corners_3d indexed by detected_ids
        let detected_corners_obj: Option<Vec<[f64; 3]>> = if self.all_charuco_corners_3d.is_empty() {
            None
        } else {
            let max_id = self.all_charuco_corners_3d.len() as i32;
            let result: Vec<[f64; 3]> = detected_ids
                .iter()
                .filter_map(|&id| {
                    if id >= 0 && id < max_id {
                        Some(self.all_charuco_corners_3d[id as usize])
                    } else {
                        None
                    }
                })
                .collect();
            if result.is_empty() { None } else { Some(result) }
        };

        // 6. Extract aruco markers (filtered to board members)
        let mut aruco_ids: Vec<i32> = Vec::new();
        let mut aruco_corners: Vec<[[f64; 2]; 4]> = Vec::new();
        let n_markers = marker_corners.len();
        for i in 0..n_markers {
            let id = marker_ids_mat
                .at_2d::<i32>(i as i32, 0).ok().copied().unwrap_or(-1);
            if id < 0 || !self.all_aruco_set.contains(&id) {
                continue;
            }
            if let Ok(inner) = marker_corners.get(i) {
                let n_c = inner.len().min(4);
                let mut c4 = [[0.0f64; 2]; 4];
                for j in 0..n_c {
                    if let Ok(pt) = inner.get(j) {
                        c4[j] = [pt.x as f64, pt.y as f64];
                    }
                }
                aruco_ids.push(id);
                aruco_corners.push(c4);
            }
        }

        // 7. Build PointCloud (full-array format)
        let points = self.build_point_cloud(&detected_ids, &detected_corners_img);
        let _t3 = std::time::Instant::now();

        let obs = CharucoObservation::new(
            frame_number,
            points,
            (image_h, image_w),
            self.all_charuco_ids.clone(),
            if self.all_charuco_corners_3d.is_empty() { None } else { Some(self.all_charuco_corners_3d.clone()) },
            self.all_aruco_ids.clone(),
            if self.all_aruco_corners_3d.is_empty() { None } else { Some(self.all_aruco_corners_3d.clone()) },
            if raw_corners.is_empty() { None } else { Some(raw_corners.clone()) },
            detected_ids,
            detected_corners_img,
            detected_corners_obj,
            aruco_ids,
            aruco_corners,
        );
        let _t4 = std::time::Instant::now();

        // ── TEMPORARY: timing breakdown ──────────────────────────────────
        let gray_us    = (_t1 - _t0).as_micros();
        let detect_us  = (_t2 - _t1).as_micros(); // detect_board: marker detection + charuco in one C++ call
        let extract_us = (_t3 - _t2).as_micros(); // result extraction + aruco filtering
        let build_us   = (_t4 - _t3).as_micros(); // CharucoObservation construction
        let total_us   = (_t4 - _t0).as_micros();
        eprintln!(
            "[charuco-detect] frame={frame_number} total={total_us}us | \
             gray={gray_us}us detect={detect_us}us \
             extract={extract_us}us build={build_us}us | \
             ids={n_ids} markers={n_markers}",
            n_ids = obs.detected_charuco_corner_ids.len(),
            n_markers = obs.detected_aruco_marker_ids.len(),
        );
        obs
    }

    fn empty_obs(&self, frame_number: u64, image_h: u32, image_w: u32) -> CharucoObservation {
        let empty_points = self.build_point_cloud(&[], &[]);
        CharucoObservation::new(
            frame_number,
            empty_points,
            (image_h, image_w),
            self.all_charuco_ids.clone(),
            if self.all_charuco_corners_3d.is_empty() { None } else { Some(self.all_charuco_corners_3d.clone()) },
            self.all_aruco_ids.clone(),
            if self.all_aruco_corners_3d.is_empty() { None } else { Some(self.all_aruco_corners_3d.clone()) },
            None,
            vec![],
            vec![],
            None,
            vec![],
            vec![],
        )
    }

    /// Run aruco marker detection on a grayscale image.
    fn build_point_cloud(&self, detected_ids: &[i32], image_coords: &[[f64; 2]]) -> PointCloud {
        let n_corners = self.corner_names.len();
        let mut xyz = ndarray::Array2::from_elem((n_corners, 3), f64::NAN);
        let mut visibility = ndarray::Array1::zeros(n_corners);

        for (&id, corner) in detected_ids.iter().zip(image_coords.iter()) {
            let idx = id as usize;
            if idx < n_corners {
                xyz[[idx, 0]] = corner[0];
                xyz[[idx, 1]] = corner[1];
                xyz[[idx, 2]] = 0.0;
                visibility[idx] = 1.0;
            }
        }

        PointCloud::new(self.corner_names.clone(), xyz, visibility)
    }

    /// Draw charuco corner markers, aruco bounding boxes, labels, and
    /// undetected corner list.  `image` is mutated in-place.
    pub fn draw_markers_into(&self, image: &mut Mat, obs: &dyn Observation) {
        let o = match obs.as_any().downcast_ref::<CharucoObservation>() {
            Some(c) => c,
            None => return,
        };

        // ── Reusable buffers (avoid per-marker heap allocations) ────────
        let mut marker_pts: Vector<opencv::core::Point> = Vector::with_capacity(4);
        let mut label = String::with_capacity(32);

        // ── Aruco marker bounding boxes (green) ───────────────────────────
        for (i, &id) in o.detected_aruco_marker_ids.iter().enumerate() {
            if let Some(corners) = o.detected_aruco_marker_corners.get(i) {
                marker_pts.clear();
                for &[x, y] in corners.iter() {
                    marker_pts.push(opencv::core::Point::new(x as i32, y as i32));
                }
                let _ = imgproc::polylines(
                    image, &marker_pts, true,
                    ARUCO_LINES_COLOR, ARUCO_LINES_THICKNESS, imgproc::LINE_8, 0,
                );
                label.clear();
                std::fmt::Write::write_fmt(&mut label, format_args!("ArUco#{}", id)).ok();
                let _ = draw_text(image, &label, corners[0][0] as i32 + 10, corners[0][1] as i32 + 10);
            }
        }

        // ── Charuco corner markers (magenta diamonds) ────────────────────
        for (i, corner) in o.detected_charuco_corners_image_coordinates.iter().enumerate() {
            let cx = corner[0] as i32;
            let cy = corner[1] as i32;
            let _ = imgproc::draw_marker(
                image, opencv::core::Point::new(cx, cy),
                CORNER_MARKER_COLOR, imgproc::MARKER_DIAMOND,
                CORNER_MARKER_SIZE, CORNER_MARKER_THICKNESS, imgproc::LINE_8,
            );
            if let Some(&id) = o.detected_charuco_corner_ids.get(i) {
                label.clear();
                std::fmt::Write::write_fmt(&mut label, format_args!("Corner#{}", id)).ok();
                let _ = draw_text(image, &label, cx + 12, cy + 12);
            }
        }

        // ── Undetected corners list ──────────────────────────────────────
        // Compute undetected corners WITHOUT cloning + O(n²) retain
        let detected_set: HashSet<i32> = o.detected_charuco_corner_ids.iter().copied().collect();
        let mut undetected: Vec<i32> = vec![];
        for &id in &o.all_charuco_ids {
            if !detected_set.contains(&id) {
                undetected.push(id);
            }
        }
        if !undetected.is_empty() {
            let panel_x = o.image_size.0 as i32 - 220;
            let _ = draw_text(image, "Undetected Corners:", panel_x, 20);
            for (i, &cid) in undetected.iter().enumerate() {
                label.clear();
                std::fmt::Write::write_fmt(&mut label, format_args!("  - {}", cid)).ok();
                let _ = draw_text(image, &label, panel_x, 40 + (i as i32) * 20);
            }
        }
    }
}

impl Tracker for CharucoTracker {
    fn process_image(&mut self, frame_number: u64, image: &Mat) -> Box<dyn Observation> {
        Box::new(self.detect(frame_number, image))
    }

    fn annotate_image(&self, image: &Mat, obs: &dyn Observation) -> Mat {
        let mut annotated = image.clone();
        self.draw_markers_into(&mut annotated, obs);
        annotated
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn format_corner_names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("CharucoCorner-{}", i)).collect()
}

fn draw_text(image: &mut Mat, text: &str, x: i32, y: i32) -> opencv::Result<()> {
    imgproc::put_text(
        image, text, opencv::core::Point::new(x, y),
        TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, imgproc::LINE_AA, false,
    )
}
