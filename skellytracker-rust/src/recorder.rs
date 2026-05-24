use std::path::Path;

use ndarray::Array3;
use serde_json::Value;

use crate::traits::Observation;

pub struct Recorder {
    observations: Vec<Box<dyn Observation>>,
}

impl Recorder {
    pub fn new() -> Self {
        Recorder {
            observations: Vec::new(),
        }
    }

    pub fn add(&mut self, obs: Box<dyn Observation>) {
        self.observations.push(obs);
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    pub fn clear(&mut self) {
        self.observations.clear();
    }

    pub fn to_array(&self) -> Option<Array3<f64>> {
        if self.observations.is_empty() {
            return None;
        }
        let n_frames = self.observations.len();
        let n_points = self.observations[0].point_cloud().n_points();
        let mut arr = Array3::zeros((n_frames, n_points, 2));
        for (f, obs) in self.observations.iter().enumerate() {
            let xy = obs.point_cloud().to_2d_array();
            arr.slice_mut(ndarray::s![f, .., ..]).assign(&xy);
        }
        Some(arr)
    }

    pub fn save_npy(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        match self.to_array() {
            Some(arr) => {
                ndarray_npy::write_npy(path, &arr)?;
                Ok(())
            }
            None => Err("No observations to save".into()),
        }
    }

    pub fn to_json_string(&self) -> String {
        let entries: Vec<String> = self
            .observations
            .iter()
            .enumerate()
            .map(|(frame, obs)| format!("\"{}\": {}", frame, obs.to_json()))
            .collect();
        format!("{{{}}}", entries.join(",\n"))
    }

    pub fn to_json_value(&self) -> Value {
        let map: serde_json::Map<String, Value> = self
            .observations
            .iter()
            .enumerate()
            .map(|(frame, obs)| {
                (
                    frame.to_string(),
                    serde_json::from_str(&obs.to_json()).unwrap_or(Value::Null),
                )
            })
            .collect();
        Value::Object(map)
    }

    pub fn save_json_file(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.to_json_string();
        std::fs::write(path, json)?;
        Ok(())
    }
}

impl Default for Recorder {
    fn default() -> Self {
        Self::new()
    }
}
