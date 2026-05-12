use std::collections::HashMap;
use ndarray::{Array1, Array2};

/// Canonical data primitive for tracked landmarks.
///
/// The i-th name in `names` corresponds to the i-th row in `xyz`
/// and the i-th element in `visibility`.
#[derive(Debug, Clone)]
pub struct PointCloud {
    pub names: Vec<String>,
    pub xyz: Array2<f64>,             // (N, 3), z=0 for 2D-only trackers
    pub visibility: Array1<f64>,       // (N,), 0.0 = not detected
    name_to_index: HashMap<String, usize>,
}

impl PointCloud {
    /// Create a new PointCloud with all-NaN coordinates and zero visibility.
    pub fn new(names: Vec<String>) -> Self {
        let number_of_points = names.len();
        let mut name_to_index = HashMap::with_capacity(number_of_points);
        for (index, name) in names.iter().enumerate() {
            name_to_index.insert(name.clone(), index);
        }
        Self {
            names,
            xyz: Array2::from_elem((number_of_points, 3), f64::NAN),
            visibility: Array1::zeros(number_of_points),
            name_to_index,
        }
    }

    /// Look up the array index for a named point. O(1).
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }

    /// Get the (x, y, z) coordinates for a named point.
    pub fn coordinates_by_name(&self, name: &str) -> Option<[f64; 3]> {
        let index = self.name_to_index.get(name)?;
        let row = self.xyz.row(*index);
        Some([row[0], row[1], row[2]])
    }

    /// Immutable view of the xy columns. Zero-copy.
    pub fn xy_view(&self) -> ndarray::ArrayView2<f64> {
        self.xyz.slice(ndarray::s![.., 0..2])
    }

    /// Number of points in this cloud.
    pub fn number_of_points(&self) -> usize {
        self.names.len()
    }

    /// Mask of points with valid (non-NaN) coordinates.
    pub fn valid_mask(&self) -> Array1<bool> {
        self.xyz.column(0).mapv(|x| !x.is_nan())
    }

    /// Count of points with valid (non-NaN) coordinates.
    pub fn number_of_valid(&self) -> usize {
        self.valid_mask().iter().filter(|&&valid| valid).count()
    }

    /// Copy out a subset of points by name. Returns a new PointCloud.
    pub fn slice_by_names(&self, requested_names: &[&str]) -> Self {
        let mut new_cloud = Self::new(
            requested_names.iter().map(|n| n.to_string()).collect(),
        );
        for (new_index, name) in requested_names.iter().enumerate() {
            if let Some(old_index) = self.name_to_index.get(*name) {
                let row = self.xyz.row(*old_index);
                new_cloud.xyz.row_mut(new_index).assign(&row);
                new_cloud.visibility[new_index] = self.visibility[*old_index];
            }
        }
        new_cloud
    }

    /// Filter points by confidence threshold. Returns a new PointCloud.
    /// If `fill_with_nans` is true, points below threshold get NaN coordinates.
    pub fn filtered_by_confidence(
        &self,
        threshold: f64,
        fill_with_nans: bool,
    ) -> Self {
        let mut filtered = self.clone();
        for index in 0..self.number_of_points() {
            if self.visibility[index] < threshold {
                filtered.visibility[index] = 0.0;
                if fill_with_nans {
                    filtered.xyz.row_mut(index).fill(f64::NAN);
                }
            }
        }
        filtered
    }

    /// Convert to a (N, 2) array of xy coordinates (always N rows, NaN fill).
    pub fn to_2d_array(&self) -> Array2<f64> {
        self.xy_view().to_owned()
    }
}
