mod types;

use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use opencv::core::{Mat, CV_8UC3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::trackers::brightest_point::BrightestPointTracker;
use crate::trackers::brightest_point::observation::BrightestPointObservation;
use crate::trackers::charuco::CharucoTracker;
use crate::trackers::charuco::observation::CharucoObservation;
use crate::trackers::rtmpose::RtmPoseTracker;
use crate::trackers::rtmpose::observation::RtmPoseObservation;
use crate::traits::{Observation, Tracker};

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
        // to_owned() does a direct memcpy (vs from_shape_fn which runs
        // a closure per pixel — 2.7M calls for a 720p frame).
        let arr = image.as_array();
        let out = arr.to_owned();
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
// CharucoTracker — Python wrapper
// ============================================================================

#[pyclass(name = "CharucoTracker")]
struct PyCharucoTracker {
    inner: std::sync::Mutex<CharucoTracker>,
    /// The most recent detection result. Stored so `annotate_image` can draw
    /// from the real Rust observation (including aruco marker corner data)
    /// instead of reconstructing from the JSON dict that crossed the FFI boundary.
    last_obs: std::sync::Mutex<Option<CharucoObservation>>,
}

#[pymethods]
impl PyCharucoTracker {
    #[new]
    fn new(
        squares_x: u32,
        squares_y: u32,
        square_length_mm: f32,
        marker_length_ratio: f32,
        dictionary_enum: i32,
    ) -> PyResult<Self> {
        match CharucoTracker::new(
            squares_x,
            squares_y,
            square_length_mm,
            marker_length_ratio,
            dictionary_enum,
        ) {
            Ok(inner) => Ok(PyCharucoTracker {
                inner: std::sync::Mutex::new(inner),
                last_obs: std::sync::Mutex::new(None),
            }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e.to_string())),
        }
    }

    #[getter]
    fn squares_x(&self) -> u32 {
        self.inner.lock().unwrap().squares_x
    }

    #[getter]
    fn squares_y(&self) -> u32 {
        self.inner.lock().unwrap().squares_y
    }

    #[getter]
    fn all_charuco_ids(&self) -> Vec<i32> {
        self.inner.lock().unwrap().all_charuco_ids.clone()
    }

    #[getter]
    fn all_aruco_ids(&self) -> Vec<i32> {
        self.inner.lock().unwrap().all_aruco_ids.clone()
    }

    /// Run charuco board detection on a numpy (H, W, 3) uint8 BGR image.
    /// Returns a dict with: frame_number, tracker_type, xy, visibility,
    /// detected_charuco_corner_ids, detected_aruco_marker_ids/corners.
    /// The full Rust observation is held internally for `annotate_image`.
    #[allow(deprecated)]
    fn process_image(
        &mut self,
        py: Python<'_>,
        frame_number: u64,
        image: PyReadonlyArrayDyn<u8>,
    ) -> PyResult<PyObject> {
        let mat = numpy_to_mat(&image)?;
        let obs = {
            let tracker = self.inner.lock().unwrap();
            tracker.detect(frame_number, &mat)
        };
        let json_str = obs.to_json();

        *self.last_obs.lock().unwrap() = Some(obs);

        let result: PyObject = py
            .import_bound("json")?
            .call_method1("loads", (json_str,))?
            .into();
        Ok(result)
    }

    /// Draw charuco corner markers, aruco bounding boxes, and labels from
    /// the most recent `process_image` result.
    ///
    /// The `observation` dict parameter is accepted for API compatibility but
    /// is **not used** for drawing — the real Rust observation stored by
    /// `process_image` is used instead.
    /// Returns a numpy (H, W, 3) uint8 BGR array.
    fn annotate_image(
        &self,
        py: Python<'_>,
        image: PyReadonlyArrayDyn<u8>,
        _observation: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyAny>> {
        let arr = image.as_array();
        let out = arr.to_owned();
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

        {
            let tracker = self.inner.lock().unwrap();
            let last_obs = self.last_obs.lock().unwrap();
            if let Some(ref obs) = *last_obs {
                tracker.draw_markers_into(&mut annotated, obs);
            }
        }

        drop(annotated);

        let bound_arr = out.into_pyarray(py);
        Ok(bound_arr.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        let tracker = self.inner.lock().unwrap();
        format!(
            "CharucoTracker(squares_x={}, squares_y={})",
            tracker.squares_x, tracker.squares_y
        )
    }
}

// ============================================================================
// RtmPoseTracker — Python wrapper (Mutex-wrapped like Charuco)
// ============================================================================

#[pyclass(name = "RtmPoseTracker")]
struct PyRtmPoseTracker {
    inner: std::sync::Mutex<RtmPoseTracker>,
    last_obs: std::sync::Mutex<Option<RtmPoseObservation>>,
}

#[pymethods]
impl PyRtmPoseTracker {
    #[new]
    fn new(mode: &str) -> PyResult<Self> {
        match RtmPoseTracker::new(mode) {
            Ok(inner) => Ok(PyRtmPoseTracker {
                inner: std::sync::Mutex::new(inner),
                last_obs: std::sync::Mutex::new(None),
            }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e.to_string())),
        }
    }

    #[getter]
    fn mode(&self) -> String {
        self.inner.lock().unwrap().mode.clone()
    }

    /// Run two-stage inference on a numpy (H, W, 3) uint8 BGR image.
    /// Returns a dict with: tracker_type, frame_number, image_size,
    /// point_names, xy, visibility, keypoints, scores.
    /// The full Rust observation is held internally for `annotate_image`.
    #[allow(deprecated)]
    fn process_image(
        &mut self,
        py: Python<'_>,
        frame_number: u64,
        image: PyReadonlyArrayDyn<u8>,
    ) -> PyResult<PyObject> {
        let mat = numpy_to_mat(&image)?;
        let obs = {
            let mut tracker = self.inner.lock().unwrap();
            tracker.detect(frame_number, &mat)
        };
        let json_str = obs.to_json();

        *self.last_obs.lock().unwrap() = Some(obs);

        let result: PyObject = py
            .import_bound("json")?
            .call_method1("loads", (json_str,))?
            .into();
        Ok(result)
    }

    /// Draw skeleton + keypoints from the most recent `process_image` result.
    /// The `observation` dict parameter is accepted for API compatibility but
    /// is NOT used for drawing — the real Rust observation is used instead.
    /// Returns a numpy (H, W, 3) uint8 BGR array.
    fn annotate_image(
        &self,
        py: Python<'_>,
        image: PyReadonlyArrayDyn<u8>,
        _observation: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyAny>> {
        let arr = image.as_array();
        let out = arr.to_owned();
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

        {
            let tracker = self.inner.lock().unwrap();
            let last_obs = self.last_obs.lock().unwrap();
            if let Some(ref obs) = *last_obs {
                tracker.draw_markers_into(&mut annotated, obs);
            }
        }

        drop(annotated);

        let bound_arr = out.into_pyarray(py);
        Ok(bound_arr.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        let tracker = self.inner.lock().unwrap();
        format!("RtmPoseTracker(mode={})", tracker.mode)
    }
}

// ============================================================================
// Python module entry point
// ============================================================================

#[pymodule]
fn _skellytracker_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBrightestPointTracker>()?;
    m.add_class::<PyCharucoTracker>()?;
    m.add_class::<PyRtmPoseTracker>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
