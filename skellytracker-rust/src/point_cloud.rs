use std::collections::HashMap;

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct PointCloud {
    pub names: Vec<String>,
    pub xyz: Array2<f64>,
    pub visibility: Array1<f64>,
    #[serde(skip)]
    name_to_idx: HashMap<String, usize>,
}

impl PointCloud {
    pub fn new(names: Vec<String>, xyz: Array2<f64>, visibility: Array1<f64>) -> Self {
        let n = names.len();
        assert_eq!(
            xyz.shape(),
            &[n, 3],
            "xyz shape {:?} must be ({}, 3)",
            xyz.shape(),
            n
        );
        assert_eq!(
            visibility.len(),
            n,
            "visibility length {} must be {}",
            visibility.len(),
            n
        );
        let name_to_idx: HashMap<String, usize> = names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        PointCloud {
            names,
            xyz,
            visibility,
            name_to_idx,
        }
    }

    pub fn empty(names: Vec<String>) -> Self {
        let n = names.len();
        PointCloud {
            name_to_idx: names
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect(),
            names,
            xyz: Array2::from_elem((n, 3), f64::NAN),
            visibility: Array1::zeros(n),
        }
    }

    pub fn n_points(&self) -> usize {
        self.names.len()
    }

    pub fn xy(&self) -> ArrayView2<'_, f64> {
        self.xyz.slice(s![.., ..2])
    }

    pub fn valid_mask(&self) -> Array1<bool> {
        self.xyz.map_axis(ndarray::Axis(1), |row| !row.iter().any(|v| v.is_nan()))
    }

    pub fn n_valid(&self) -> usize {
        self.valid_mask().iter().filter(|&&v| v).count()
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.name_to_idx.get(name).copied()
    }

    pub fn has_name(&self, name: &str) -> bool {
        self.name_to_idx.contains_key(name)
    }

    pub fn xyz_by_name(&self, name: &str) -> Option<ArrayView1<'_, f64>> {
        let idx = self.name_to_idx.get(name)?;
        Some(self.xyz.row(*idx))
    }

    pub fn xy_by_name(&self, name: &str) -> Option<Array1<f64>> {
        self.xyz_by_name(name)
            .map(|row| row.slice(s![..2]).to_owned())
    }

    pub fn to_2d_array(&self) -> Array2<f64> {
        self.xy().to_owned()
    }

    pub fn to_named_dict(&self) -> HashMap<String, Array1<f64>> {
        self.names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), self.xyz.row(i).slice(s![..2]).to_owned()))
            .collect()
    }

    pub fn to_valid_dict(&self) -> HashMap<String, Array1<f64>> {
        let mask = self.valid_mask();
        self.names
            .iter()
            .enumerate()
            .filter(|(i, _)| mask[*i])
            .map(|(i, name)| (name.clone(), self.xyz.row(i).slice(s![..2]).to_owned()))
            .collect()
    }

    pub fn filtered_by_confidence(&self, threshold: f64) -> PointCloud {
        let mut xyz = self.xyz.clone();
        let vis = self.visibility.clone();
        for i in 0..self.n_points() {
            if vis[i] < threshold {
                xyz.row_mut(i).fill(f64::NAN);
            }
        }
        PointCloud::new(self.names.clone(), xyz, vis)
    }

    pub fn concatenate(clouds: &[&PointCloud]) -> PointCloud {
        assert!(!clouds.is_empty(), "Cannot concatenate empty list");
        let mut all_names = Vec::new();
        let mut all_xyz_parts = Vec::new();
        let mut all_vis_parts = Vec::new();
        for cloud in clouds {
            all_names.extend(cloud.names.clone());
            all_xyz_parts.push(cloud.xyz.view());
            all_vis_parts.push(cloud.visibility.view());
        }
        let n_total: usize = all_names.len();
        let mut xyz = Array2::zeros((n_total, 3));
        let mut vis = Array1::zeros(n_total);
        let mut offset = 0;
        for (cloud_xyz, cloud_vis) in all_xyz_parts.iter().zip(all_vis_parts.iter()) {
            let n = cloud_xyz.shape()[0];
            xyz.slice_mut(s![offset..offset + n, ..]).assign(cloud_xyz);
            vis.slice_mut(s![offset..offset + n]).assign(cloud_vis);
            offset += n;
        }
        PointCloud::new(all_names, xyz, vis)
    }

    pub fn slice_by_names(&self, names: &[String]) -> PointCloud {
        let indices: Vec<usize> = names
            .iter()
            .map(|n| self.name_to_idx[n.as_str()])
            .collect();
        let xyz = self.xyz.select(ndarray::Axis(0), &indices);
        let vis = self.visibility.select(ndarray::Axis(0), &indices);
        PointCloud::new(names.to_vec(), xyz, vis)
    }
}
