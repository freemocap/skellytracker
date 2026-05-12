use std::path::Path;

use anyhow::Result;
use ndarray::Array3;
use npyz::WriterBuilder;

use crate::core::observation::Observation;
use crate::core::traits::Record;

/// Collects observations across frames and serializes to JSON or .npy.
pub struct Recorder {
    observations: Vec<Observation>,
}

impl Recorder {
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
        }
    }
}

impl Default for Recorder {
    fn default() -> Self {
        Self::new()
    }
}

impl Record for Recorder {
    fn add_observation(&mut self, observation: Observation) {
        self.observations.push(observation);
    }

    fn clear(&mut self) {
        self.observations.clear();
    }

    fn observation_count(&self) -> usize {
        self.observations.len()
    }

    fn to_json_string(&self) -> Result<String> {
        let value = serde_json::json!({
            "observations": self.observations.iter().map(|observation| {
                serde_json::json!({
                    "frame_number": observation.frame_number,
                    "tracker_kind": format!("{:?}", observation.tracker_kind),
                    "number_of_valid_points": observation.points.number_of_valid(),
                })
            }).collect::<Vec<_>>(),
        });
        Ok(serde_json::to_string_pretty(&value)?)
    }

    fn save_npy(&self, path: &Path) -> Result<()> {
        if self.observations.is_empty() {
            return Ok(());
        }
        let frame_count = self.observations.len();
        let point_count = self.observations[0].points.number_of_points();
        let mut stacked = Array3::from_elem((frame_count, point_count, 2), f64::NAN);
        for (frame_index, observation) in self.observations.iter().enumerate() {
            let xy = observation.points.xy_view();
            for point_index in 0..point_count {
                stacked[[frame_index, point_index, 0]] = xy[[point_index, 0]];
                stacked[[frame_index, point_index, 1]] = xy[[point_index, 1]];
            }
        }
        let shape = stacked.shape();
        let flat: Vec<f64> = stacked.iter().copied().collect();
        let file = std::fs::File::create(path)?;
        let mut writer = npyz::WriteOptions::<f64>::new()
            .default_dtype()
            .shape(&[shape[0] as u64, shape[1] as u64, shape[2] as u64])
            .writer(file)
            .begin_nd()?;
        writer.extend(flat.iter().copied())?;
        writer.finish()?;
        Ok(())
    }
}

impl Recorder {
    /// Stack all PointCloud xy arrays into (frames, points, 2) array.
    pub fn to_stacked_array(&self) -> Option<Array3<f64>> {
        if self.observations.is_empty() {
            return None;
        }
        let frame_count = self.observations.len();
        let point_count = self.observations[0].points.number_of_points();
        let mut stacked = Array3::from_elem((frame_count, point_count, 2), f64::NAN);
        for (frame_index, observation) in self.observations.iter().enumerate() {
            let xy = observation.points.xy_view();
            for point_index in 0..point_count {
                stacked[[frame_index, point_index, 0]] = xy[[point_index, 0]];
                stacked[[frame_index, point_index, 1]] = xy[[point_index, 1]];
            }
        }
        Some(stacked)
    }
}
