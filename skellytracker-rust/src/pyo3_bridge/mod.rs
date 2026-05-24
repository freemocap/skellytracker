mod types;

use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use opencv::core::{Mat, CV_8UC3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::trackers::brightest_point::BrightestPointTracker;
use crate::trackers::brightest_point::observation::BrightestPointObservation;
use crate::traits::Tracker;

/// Build an OpenCV Mat header that borrows a Python numpy uint8 array.
/// The Mat does NOT own the data — the numpy array must outlive the Mat.
fn numpy_to_mat(arr: &PyReadonlyArrayDyn<u8>) -> PyResult<Mat> {
    let view = arr.as_array();
    let shape = view.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected array of shape (H, W, 3), got {:?}",
            shape
        )));
    }
    let slice = arr.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {e}"))
    })?;
    let data_ptr = slice.as_ptr() as *mut std::ffi::c_void;
    unsafe {
        Ok(Mat::new_rows_cols_with_data_unsafe_def(
            shape[0] as i32,
            shape[1] as i32,
            CV_8UC3,
            data_ptr,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Mat::new: {e}")))?)
    }
}

// ============================================================================
// BrightestPointTracker — Python wrapper
// ============================================================================

#[pyclass(name = "BrightestPointTracker")]
struct PyBrightestPointTracker {
    inner: BrightestPointTracker,
    /// The most recent detection result.  Stored so `annotate_image` can
    /// draw directly from the real Rust observation (including contour
    /// data) instead of reconstructing a degraded copy from the Python
    /// dict that crossed the FFI boundary.
    last_obs: Option<BrightestPointObservation>,
}

#[pymethods]
impl PyBrightestPointTracker {
    #[new]
    fn new(num_points: usize, luminance_threshold: u8) -> Self {
        PyBrightestPointTracker {
            inner: BrightestPointTracker::new(num_points, luminance_threshold),
            last_obs: None,
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
    /// The full Rust observation (with contour data) is held internally for
    /// `annotate_image`.
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

        // Stash the concrete observation so annotate_image has access to
        // the full detection data (including contour polygons for outlines).
        self.last_obs = obs
            .as_any()
            .downcast_ref::<BrightestPointObservation>()
            .cloned();

        let result: PyObject = py
            .import_bound("json")?
            .call_method1("loads", (json_str,))?
            .into();
        Ok(result)
    }

    /// Draw cross markers (and blob outlines when contour data is available)
    /// from the most recent `process_image` result.
    ///
    /// The `observation` dict parameter is accepted for API compatibility with
    /// the Python-side caller but is **not used** for drawing — the real Rust
    /// observation stored by `process_image` is used instead, so annotation
    /// has access to the full detection data including contour polygons.
    /// Returns a numpy (H, W, 3) uint8 BGR array.
    fn annotate_image(
        &self,
        py: Python<'_>,
        image: PyReadonlyArrayDyn<u8>,
        _observation: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyAny>> {
        // Single copy: numpy array we own and can draw into.
        let arr = image.as_array();
        let out = ndarray::Array3::<u8>::from_shape_fn(
            (arr.shape()[0], arr.shape()[1], 3),
            |(y, x, c)| arr[[y, x, c]],
        );
        let out_ptr = out.as_ptr() as *mut std::ffi::c_void;

        let mut annotated = unsafe {
            Mat::new_rows_cols_with_data_unsafe_def(
                arr.shape()[0] as i32,
                arr.shape()[1] as i32,
                CV_8UC3,
                out_ptr,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Mat::new: {e}")))?
        };

        if let Some(ref obs) = self.last_obs {
            self.inner.draw_markers_into(&mut annotated, obs);
        }

        drop(annotated); // release Mat borrow before moving out

        let bound_arr = out.into_pyarray(py);
        Ok(bound_arr.into_any().unbind())
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
