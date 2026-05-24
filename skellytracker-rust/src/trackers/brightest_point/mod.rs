pub mod observation;

use opencv::core::{Mat, Point, Scalar, Vector};
use opencv::imgproc;
use observation::{BrightPatch, BrightestPointObservation};
use crate::point_cloud::PointCloud;
use crate::traits::{Observation, Tracker};

// Drawing constants — must match the Python BrightestPointTracker annotate_image.
const MARKER_COLOR: Scalar = Scalar::new(0.0, 0.0, 255.0, 0.0); // BGR red
const MARKER_TYPE: i32 = imgproc::MARKER_CROSS;
const MARKER_SIZE: i32 = 20;
const MARKER_THICKNESS: i32 = 2;
const MARKER_LINE_TYPE: i32 = imgproc::LINE_8;

// Blob outline drawing.
const OUTLINE_COLOR: Scalar = Scalar::new(255.0, 0.0, 255.0, 0.0); // BGR magenta
const OUTLINE_THICKNESS: i32 = 2;

pub struct BrightestPointTracker {
    pub num_points: usize,
    pub luminance_threshold: f64,
}

impl BrightestPointTracker {
    pub fn new(num_points: usize, luminance_threshold: u8) -> Self {
        BrightestPointTracker {
            num_points,
            luminance_threshold: luminance_threshold as f64,
        }
    }

    pub fn find_bright_patches(&self, gray: &Mat) -> Vec<BrightPatch> {
        let mut thresholded = Mat::default();
        if imgproc::threshold(
            gray,
            &mut thresholded,
            self.luminance_threshold,
            255.0,
            imgproc::THRESH_BINARY,
        )
        .is_err()
        {
            eprintln!("[skellytracker-rust] threshold failed — returning 0 patches");
            return Vec::new();
        }

        let mut contours: Vector<Vector<Point>> = Vector::new();
        if imgproc::find_contours(
            &thresholded,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point::default(),
        )
        .is_err()
        {
            eprintln!("[skellytracker-rust] find_contours failed — returning 0 patches");
            return Vec::new();
        }

        let mut patches: Vec<BrightPatch> = contours
            .iter()
            .filter_map(|contour| {
                let moments = imgproc::moments(&contour, false).ok()?;
                let area = imgproc::contour_area(&contour, false).ok()?;
                if moments.m00 == 0.0 || area <= 0.0 {
                    return None;
                }
                let pts: Vec<Point> = contour.iter().collect();
                Some(BrightPatch {
                    area,
                    centroid_x: (moments.m10 / moments.m00) as i32,
                    centroid_y: (moments.m01 / moments.m00) as i32,
                    contour: pts,
                })
            })
            .collect();

        patches.sort_by(|a, b| {
            b.area
                .partial_cmp(&a.area)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        patches.truncate(self.num_points);

        patches
    }

    pub fn build_point_cloud(&self, patches: &[BrightPatch]) -> PointCloud {
        let n = self.num_points;
        let names: Vec<String> = (0..n).map(|i| format!("brightest_point_{i}")).collect();
        let mut xyz = ndarray::Array2::from_elem((n, 3), f64::NAN);
        let mut visibility = ndarray::Array1::zeros(n);

        for (i, patch) in patches.iter().enumerate() {
            xyz[[i, 0]] = patch.centroid_x as f64;
            xyz[[i, 1]] = patch.centroid_y as f64;
            xyz[[i, 2]] = 0.0;
            visibility[i] = 1.0;
        }

        PointCloud::new(names, xyz, visibility)
    }

    /// Draw blob outlines (from stored contour data) and cross markers at
    /// every visible point centroid.  `image` is mutated in-place.
    pub fn draw_markers_into(&self, image: &mut Mat, obs: &dyn Observation) {
        // ── Blob outlines (drawn first so markers render on top) ─────────
        if let Some(bp) = obs
            .as_any()
            .downcast_ref::<observation::BrightestPointObservation>()
        {
            for patch in &bp.patches {
                if patch.contour.len() < 3 {
                    continue;
                }
                let pts: Vector<Point> = patch.contour.iter().copied().collect();
                let _ = imgproc::polylines(
                    image,
                    &pts,
                    true, // closed polygon
                    OUTLINE_COLOR,
                    OUTLINE_THICKNESS,
                    imgproc::LINE_8,
                    0,
                );
            }
        }

        // ── Cross markers at centroids ──────────────────────────────────
        let pc = obs.point_cloud();
        let xy = pc.xy();

        for i in 0..pc.n_points() {
            let vis = pc.visibility[i];
            if vis <= 0.0 || xy[[i, 0]].is_nan() || xy[[i, 1]].is_nan() {
                continue;
            }
            let _ = imgproc::draw_marker(
                image,
                Point::new(xy[[i, 0]] as i32, xy[[i, 1]] as i32),
                MARKER_COLOR,
                MARKER_TYPE,
                MARKER_SIZE,
                MARKER_THICKNESS,
                MARKER_LINE_TYPE,
            );
        }
    }
}

impl Tracker for BrightestPointTracker {
    fn process_image(&mut self, frame_number: u64, image: &Mat) -> Box<dyn Observation> {
        let mut gray = Mat::default();
        if imgproc::cvt_color(
            image,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .is_err()
        {
            eprintln!("[skellytracker-rust] cvt_color failed — returning empty observation");
            let empty_points = self.build_point_cloud(&[]);
            return Box::new(BrightestPointObservation::new(frame_number, empty_points, vec![]));
        }

        let patches = self.find_bright_patches(&gray);
        let points = self.build_point_cloud(&patches);

        Box::new(BrightestPointObservation::new(frame_number, points, patches))
    }

    fn annotate_image(&self, image: &Mat, obs: &dyn Observation) -> Mat {
        let mut annotated = image.clone();
        self.draw_markers_into(&mut annotated, obs);
        annotated
    }
}
