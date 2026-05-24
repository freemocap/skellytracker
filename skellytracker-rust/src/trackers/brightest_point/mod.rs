pub mod observation;

use opencv::core::{Mat, Scalar, Vector};
use opencv::imgproc;


use observation::{BrightPatch, BrightestPointObservation};
use crate::point_cloud::PointCloud;
use crate::traits::{Observation, Tracker};

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
        imgproc::threshold(
            gray,
            &mut thresholded,
            self.luminance_threshold,
            255.0,
            imgproc::THRESH_BINARY,
        )
        .unwrap_or_default();

        let mut contours: Vector<Vector<opencv::core::Point>> = Vector::new();
        imgproc::find_contours(
            &thresholded,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            opencv::core::Point::default(),
        )
        .unwrap_or_default();

        let mut patches: Vec<BrightPatch> = contours
            .iter()
            .filter_map(|contour| {
                let moments = imgproc::moments(&contour, false).ok()?;
                let area = imgproc::contour_area(&contour, false).ok()?;
                if moments.m00 == 0.0 || area <= 0.0 {
                    return None;
                }
                let cx = (moments.m10 / moments.m00) as i32;
                let cy = (moments.m01 / moments.m00) as i32;
                Some(BrightPatch {
                    area,
                    centroid_x: cx,
                    centroid_y: cy,
                })
            })
            .collect();

        patches.sort_by(|a, b| b.area.partial_cmp(&a.area).unwrap_or(std::cmp::Ordering::Equal));
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
}

impl Tracker for BrightestPointTracker {
    fn process_image(&mut self, frame_number: u64, image: &Mat) -> Box<dyn Observation> {
        let mut gray = Mat::default();
        imgproc::cvt_color(
            image,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .unwrap_or_default();

        let patches = self.find_bright_patches(&gray);
        let points = self.build_point_cloud(&patches);

        Box::new(BrightestPointObservation::new(
            frame_number,
            points,
            patches,
        ))
    }

    fn annotate_image(&self, image: &Mat, obs: &dyn Observation) -> Mat {
        let mut annotated = image.clone();
        let pc = obs.point_cloud();
        let xy = pc.xy();

        for i in 0..pc.n_points() {
            let vis = pc.visibility[i];
            if vis <= 0.0 || xy[[i, 0]].is_nan() || xy[[i, 1]].is_nan() {
                continue;
            }
            let px = xy[[i, 0]] as i32;
            let py = xy[[i, 1]] as i32;

            let _ = imgproc::draw_marker(
                &mut annotated,
                opencv::core::Point::new(px, py),
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                imgproc::MARKER_CROSS,
                20,
                2,
                imgproc::LINE_8,
            );
        }
        annotated
    }
}
