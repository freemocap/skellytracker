//! MediaPipe tracker — reverse PyO3 bridge calling Python mediapipe for inference.
//!
//! Phase 1 (thin black-box wrapper): Rust holds PyObject refs to the Python
//! MediapipeCompositeDetector and MediapipeCompositeAnnotator. detect() calls
//! the Python detector via PyO3, extracts the resulting PointCloud data, and
//! wraps it in a Rust MediaPipeObservation. draw_markers_into() delegates to
//! the Python annotator.

pub mod observation;

use std::sync::Mutex;

use ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use opencv::core::Mat;
use opencv::prelude::*;
use pyo3::prelude::*;

use crate::trackers::mediapipe::observation::MediaPipeObservation;
use crate::traits::{Observation, Tracker};

pub struct MediaPipeTracker {
    /// Python MediapipeCompositeDetector instance (created by Python, passed via #[new])
    detector: Py<PyAny>,
    /// Python MediapipeCompositeAnnotator instance (created by Python, passed via #[new])
    annotator: Py<PyAny>,
    /// The Python observation object from the most recent detect() call.
    /// Stored so draw_markers_into() can pass it to the Python annotator.
    last_python_obs: Mutex<Option<Py<PyAny>>>,
}

impl MediaPipeTracker {
    /// Create from pre-constructed Python detector and annotator objects.
    ///
    /// Called from the pyclass `#[new]` — the Python adapter creates the
    /// Python-side objects (resolving model paths, building configs) and
    /// hands them across the bridge.
    pub fn new(detector: Py<PyAny>, annotator: Py<PyAny>) -> Self {
        Self {
            detector,
            annotator,
            last_python_obs: Mutex::new(None),
        }
    }

    /// Run the full composite detection pipeline by delegating to Python.
    ///
    /// Converts the BGR Mat → numpy → Python detector.detect() →
    /// extracts PointCloud data → returns Rust MediaPipeObservation.
    pub fn detect(
        &self,
        py: Python<'_>,
        frame_number: u64,
        image: &Mat,
    ) -> MediaPipeObservation {
        let image_h = image.rows() as u32;
        let image_w = image.cols() as u32;
        let image_size = (image_h, image_w);

        // Convert Mat → numpy uint8 (H, W, 3) for Python
        let numpy_image = match mat_to_numpy(py, image) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("[skellytracker-rust] MediaPipe: Mat→numpy failed: {e}");
                return MediaPipeObservation::empty(frame_number, image_size);
            }
        };

        // Call Python detector.detect(frame_number, numpy_image)
        let py_obs = match self.detector.call_method1(
            py,
            "detect",
            (frame_number, numpy_image),
        ) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("[skellytracker-rust] MediaPipe: Python detect() failed: {e}");
                return MediaPipeObservation::empty(frame_number, image_size);
            }
        };

        // Extract PointCloud data from the Python observation
        let bound_obs = py_obs.bind(py);
        let result = extract_observation_data(&bound_obs, frame_number, image_size);

        // Stash the Python observation for annotate_image to use
        *self.last_python_obs.lock().unwrap() = Some(py_obs);

        result
    }

    /// Draw skeleton, hands, face mesh onto the image by delegating to the
    /// Python annotator.
    pub fn draw_markers_into(
        &self,
        py: Python<'_>,
        image: &mut Mat,
    ) {
        let py_obs = match self.last_python_obs.lock().unwrap().as_ref() {
            Some(o) => o.clone_ref(py),
            None => return,
        };

        // Convert output Mat → numpy for Python annotator
        let numpy_image = match mat_to_numpy(py, image) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("[skellytracker-rust] MediaPipe: annotate Mat→numpy failed: {e}");
                return;
            }
        };

        // Call Python annotator.annotate_image(numpy_image, python_obs)
        let annotated_py = match self.annotator.call_method1(
            py,
            "annotate_image",
            (numpy_image, py_obs),
        ) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("[skellytracker-rust] MediaPipe: Python annotate_image() failed: {e}");
                return;
            }
        };

        // Copy annotated numpy data back into the output Mat
        let bound_annotated = annotated_py.bind(py);
        if let Err(e) = numpy_to_mat_mut(&bound_annotated, image) {
            eprintln!("[skellytracker-rust] MediaPipe: numpy→Mat copy failed: {e}");
        }
    }
}

impl Tracker for MediaPipeTracker {
    fn process_image(&mut self, frame_number: u64, _image: &Mat) -> Box<dyn Observation> {
        // This path is NOT used by the pyclass bridge — process_image is
        // overridden in the #[pymethods] to get access to Python<'py>.
        // Fallback: return an empty observation.
        Box::new(MediaPipeObservation::empty(frame_number, (0, 0)))
    }

    fn annotate_image(&self, _image: &Mat, _obs: &dyn Observation) -> Mat {
        // Same — annotation goes through the pyclass bridge.
        Mat::default()
    }
}

// ---------------------------------------------------------------------------
// numpy ↔ Mat conversion helpers (PyO3 side)
// ---------------------------------------------------------------------------

fn mat_to_numpy<'py>(py: Python<'py>, mat: &Mat) -> PyResult<Py<PyAny>> {
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    let ch = mat.channels() as usize;
    let step = mat.mat_step()[0] as usize;

    let mut flat = vec![0u8; rows * cols * ch];
    unsafe {
        let ptr = mat.data() as *const u8;
        for r in 0..rows {
            let src = std::slice::from_raw_parts(ptr.add(r * step), cols * ch);
            let dst = &mut flat[r * cols * ch..(r + 1) * cols * ch];
            dst.copy_from_slice(src);
        }
    }

    let arr = ndarray::Array3::from_shape_vec((rows, cols, ch), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("shape error: {e}")))?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

fn numpy_to_mat_mut(arr_py: &Bound<'_, PyAny>, dst: &mut Mat) -> PyResult<()> {
    let arr: PyReadonlyArrayDyn<u8> = arr_py.extract()?;
    let view = arr.as_array();
    let shape = view.shape();

    if shape.len() != 3 || shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected (H, W, 3), got {:?}", shape
        )));
    }

    let slice = arr.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("not contiguous: {e}"))
    })?;

    let dst_step = dst.mat_step()[0] as usize;
    let dst_rows = dst.rows() as usize;
    let dst_cols = dst.cols() as usize;
    let dst_ch = dst.channels() as usize;

    if shape[0] != dst_rows || shape[1] != dst_cols || shape[2] != dst_ch {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Size mismatch: numpy {:?} vs Mat {}x{}x{}",
            shape, dst_rows, dst_cols, dst_ch
        )));
    }

    unsafe {
        let dst_ptr = dst.data_mut() as *mut u8;
        for r in 0..dst_rows {
            let dst_row = std::slice::from_raw_parts_mut(dst_ptr.add(r * dst_step), dst_cols * dst_ch);
            let src_row = &slice[r * dst_cols * dst_ch..(r + 1) * dst_cols * dst_ch];
            dst_row.copy_from_slice(src_row);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Data extraction from Python observation
// ---------------------------------------------------------------------------

fn extract_observation_data(
    py_obs: &Bound<'_, PyAny>,
    frame_number: u64,
    image_size: (u32, u32),
) -> MediaPipeObservation {
    let empty = || MediaPipeObservation::empty(frame_number, image_size);

    let points = match py_obs.getattr("points") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[skellytracker-rust] MediaPipe: obs.points missing: {e}");
            return empty();
        }
    };

    // Extract names: tuple[str, ...] → Vec<String>
    let names: Vec<String> = match points.getattr("names") {
        Ok(n) => match n.extract::<Vec<String>>() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[skellytracker-rust] MediaPipe: names extract failed: {e}");
                return empty();
            }
        },
        Err(e) => {
            eprintln!("[skellytracker-rust] MediaPipe: points.names missing: {e}");
            return empty();
        }
    };

    // Extract xyz: numpy (N, 3) → Array2<f64>
    let xyz: Array2<f64> = match points.getattr("xyz") {
        Ok(arr) => match extract_numpy_f64(&arr) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("[skellytracker-rust] MediaPipe: xyz extract failed: {e}");
                return empty();
            }
        },
        Err(e) => {
            eprintln!("[skellytracker-rust] MediaPipe: points.xyz missing: {e}");
            return empty();
        }
    };

    // Extract visibility: numpy (N,) → Array1<f64>
    let visibility: ndarray::Array1<f64> = match points.getattr("visibility") {
        Ok(arr) => match extract_numpy_f64_1d(&arr) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("[skellytracker-rust] MediaPipe: vis extract failed: {e}");
                ndarray::Array1::zeros(names.len())
            }
        },
        Err(_) => ndarray::Array1::zeros(names.len()),
    };

    // Detection flags
    let has_pose = py_obs.getattr("has_pose")
        .and_then(|v| v.extract::<bool>())
        .unwrap_or(false);
    let has_right_hand = py_obs.getattr("has_right_hand")
        .and_then(|v| v.extract::<bool>())
        .unwrap_or(false);
    let has_left_hand = py_obs.getattr("has_left_hand")
        .and_then(|v| v.extract::<bool>())
        .unwrap_or(false);
    let has_face = py_obs.getattr("has_face")
        .and_then(|v| v.extract::<bool>())
        .unwrap_or(false);

    MediaPipeObservation::build(
        frame_number,
        image_size,
        names,
        xyz,
        visibility,
        has_pose,
        has_right_hand,
        has_left_hand,
        has_face,
    )
}

/// Extract a 2D float64 numpy array → ndarray Array2<f64>
fn extract_numpy_f64(arr: &Bound<'_, PyAny>) -> PyResult<Array2<f64>> {
    let readonly: PyReadonlyArrayDyn<f64> = arr.extract()?;
    let view = readonly.as_array();
    let shape = view.shape().to_vec();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected 2D array, got {:?}", shape
        )));
    }
    Ok(view.to_owned().into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("dims: {e:?}")))?)
}

/// Extract a 1D float64 numpy array → ndarray Array1<f64>
fn extract_numpy_f64_1d(arr: &Bound<'_, PyAny>) -> PyResult<ndarray::Array1<f64>> {
    let readonly: PyReadonlyArrayDyn<f64> = arr.extract()?;
    let view = readonly.as_array();
    let shape = view.shape().to_vec();
    if shape.len() != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected 1D array, got {:?}", shape
        )));
    }
    Ok(view.to_owned().into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("dims: {e:?}")))?)
}
