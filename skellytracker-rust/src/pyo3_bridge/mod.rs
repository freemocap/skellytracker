mod types;

use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use opencv::core::{Mat, MatTraitConst, Point, Scalar, CV_8UC3};
use opencv::imgproc;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::trackers::brightest_point::BrightestPointTracker;
use crate::traits::Tracker;

/// Build an OpenCV Mat that borrows a Python numpy uint8 array.
fn numpy_to_mat(arr: &PyReadonlyArrayDyn<u8>) -> PyResult<Mat> {
    let view = arr.as_array();
    let shape = view.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected array of shape (H, W, 3), got {:?}",
            shape
        )));
    }
    let rows = shape[0] as i32;
    let cols = shape[1] as i32;
    let slice = arr.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {e}"))
    })?;
    let data_ptr = slice.as_ptr() as *mut std::ffi::c_void;
    unsafe {
        Ok(Mat::new_rows_cols_with_data_unsafe_def(
            rows,
            cols,
            CV_8UC3,
            data_ptr,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Mat::new: {e}")))?)
    }
}

/// Copy an OpenCV Mat into a freshly-allocated numpy (H, W, 3) uint8 array.
fn mat_to_numpy(py: Python<'_>, mat: &Mat) -> PyResult<Py<PyAny>> {
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    let channels = mat.channels() as usize;
    let total = rows * cols * channels;
    let data_ptr = mat.data();
    let data = unsafe { std::slice::from_raw_parts(data_ptr, total) };
    let owned = ndarray::Array3::from_shape_vec((rows, cols, channels), data.to_vec())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}")))?;
    let bound_arr = owned.into_pyarray(py);
    Ok(bound_arr.into_any().unbind())
}

// ============================================================================
// BrightestPointTracker — Python wrapper
// ============================================================================

#[pyclass(name = "BrightestPointTracker")]
struct PyBrightestPointTracker {
    inner: BrightestPointTracker,
}

#[pymethods]
impl PyBrightestPointTracker {
    #[new]
    fn new(num_points: usize, luminance_threshold: u8) -> Self {
        PyBrightestPointTracker {
            inner: BrightestPointTracker::new(num_points, luminance_threshold),
        }
    }

    #[getter]
    fn num_points(&self) -> usize {
        self.inner.num_points
    }

    #[getter]
    fn luminance_threshold(&self) -> u8 {
        self.inner.luminance_threshold as u8
    }

    /// Run detection on a numpy (H, W, 3) uint8 BGR image.
    /// Returns a dict with keys: frame_number, tracker_type, point_names, xy, visibility.
    #[allow(deprecated)]
    fn process_image(
        &mut self,
        py: Python<'_>,
        frame_number: u64,
        image: PyReadonlyArrayDyn<u8>,
    ) -> PyResult<PyObject> {
        let mat = numpy_to_mat(&image)?;
        let obs = self.inner.process_image(frame_number, &mat);
        let json_str = obs.to_json();
        let result: PyObject = py
            .import_bound("json")?
            .call_method1("loads", (json_str,))?
            .into();
        Ok(result)
    }

    /// Draw cross markers from a previous `process_image` result.
    /// `observation` must be the dict returned by `process_image`.
    /// Returns a numpy (H, W, 3) uint8 BGR array.
    fn annotate_image(
        &self,
        py: Python<'_>,
        image: PyReadonlyArrayDyn<u8>,
        observation: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyAny>> {
        let mat = numpy_to_mat(&image)?;
        let mut annotated = mat.clone();

        let xy_list = observation
            .get_item("xy")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'xy'"))?;
        let vis_list = observation
            .get_item("visibility")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'visibility'"))?;

        let xy: Vec<Vec<f64>> = xy_list.extract().map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("xy not list of [f64, f64]: {e}"))
        })?;
        let vis: Vec<f64> = vis_list.extract().map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("visibility not list of f64: {e}"))
        })?;

        for (i, coords) in xy.iter().enumerate() {
            if i >= vis.len() || vis[i] <= 0.0 {
                continue;
            }
            if coords.len() < 2 || coords[0].is_nan() || coords[1].is_nan() {
                continue;
            }
            let px = coords[0] as i32;
            let py = coords[1] as i32;
            let _ = imgproc::draw_marker(
                &mut annotated,
                Point::new(px, py),
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                imgproc::MARKER_CROSS,
                20,
                2,
                imgproc::LINE_8,
            );
        }
        mat_to_numpy(py, &annotated)
    }

    fn __repr__(&self) -> String {
        format!(
            "BrightestPointTracker(num_points={}, luminance_threshold={})",
            self.inner.num_points, self.inner.luminance_threshold
        )
    }
}

// ============================================================================
// Python module entry point
// ============================================================================

#[pymodule]
fn _skellytracker_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBrightestPointTracker>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
