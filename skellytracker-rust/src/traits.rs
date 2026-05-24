use std::any::Any;

use opencv::core::Mat;

use crate::point_cloud::PointCloud;

pub trait Observation: Send + Any {
    fn frame_number(&self) -> u64;
    fn point_cloud(&self) -> &PointCloud;
    fn to_json(&self) -> String;
    fn to_json_bytes(&self) -> Vec<u8> {
        self.to_json().into_bytes()
    }
    fn as_any(&self) -> &dyn Any;
}

pub trait Tracker {
    fn process_image(&mut self, frame_number: u64, image: &Mat) -> Box<dyn Observation>;
    fn annotate_image(&self, image: &Mat, obs: &dyn Observation) -> Mat;
}

pub trait Detector {
    fn detect(&self, frame_number: u64, image: &Mat) -> Box<dyn Observation>;
}

pub trait Annotator {
    fn annotate(&self, image: &Mat, obs: &dyn Observation) -> Mat;
}
