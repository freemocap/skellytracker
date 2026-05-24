use std::any::Any;

use crate::point_cloud::PointCloud;
use crate::traits::Observation;

#[derive(Debug, Clone)]
pub struct BrightPatch {
    pub area: f64,
    pub centroid_x: i32,
    pub centroid_y: i32,
}

#[derive(Debug, Clone)]
pub struct BrightestPointObservation {
    pub frame_number: u64,
    pub points: PointCloud,
    pub patches: Vec<BrightPatch>,
}

impl BrightestPointObservation {
    pub fn new(
        frame_number: u64,
        points: PointCloud,
        patches: Vec<BrightPatch>,
    ) -> Self {
        BrightestPointObservation {
            frame_number,
            points,
            patches,
        }
    }
}

impl Observation for BrightestPointObservation {
    fn frame_number(&self) -> u64 {
        self.frame_number
    }

    fn point_cloud(&self) -> &PointCloud {
        &self.points
    }

    fn to_json(&self) -> String {
        let point_names: Vec<&str> = self.points.names.iter().map(|s| s.as_str()).collect();
        let xy = self.points.to_2d_array();
        let vis: Vec<f64> = self.points.visibility.to_vec();

        let xy_json: Vec<Vec<f64>> = xy
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();

        serde_json::json!({
            "frame_number": self.frame_number,
            "tracker_type": "brightest_point",
            "point_names": point_names,
            "xy": xy_json,
            "visibility": vis,
        })
        .to_string()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
