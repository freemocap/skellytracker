# MediaPipe Tracker — Reverse PyO3 Bridge (Phase 1: Black-Box Wrapper)

> Fourth tracker translated after BrightestPoint, Charuco, and RTMPose. The first to use a *reverse* PyO3 bridge — Rust holds `Py<PyAny>` refs to Python `mediapipe` objects and calls them via PyO3 for inference. No Rust MediaPipe bindings exist; this hybrid approach delegates ML inference to the battle-tested Python/C++ TFLite stack while Rust owns the data model and trait implementation.

## What was translated

The Python `MediapipeCompositeDetector` and `MediapipeCompositeAnnotator` are wrapped as opaque Python objects. Rust calls them via PyO3 and builds a native Rust `MediaPipeObservation` from the returned data.

| Python file | Rust equivalent |
|-------------|-----------------|
| `composite/mediapipe_composite_detector.py` | `src/trackers/mediapipe/mod.rs` — `detect()` calls Python detector via PyO3 |
| `composite/mediapipe_composite_observation.py` | `src/trackers/mediapipe/observation.rs` — 211-point data model |
| `composite/mediapipe_composite_annotator.py` | `src/trackers/mediapipe/mod.rs` — `draw_markers_into()` delegates to Python annotator |
| `composite/composite_tracker_mappings.py` | NOT ported (fusion stays in Python for Phase 1) |
| `body/`, `hands/`, `face/` sub-detectors | NOT ported (called indirectly via composite detector) |

## Architecture — the reverse bridge

Every previous translation used this pattern:

```
Python ──calls──→ Rust (via PyO3)
  │                 │
  │  process_image  │  detect() → Box<dyn Observation>
  │  annotate_image │  draw_markers_into()
  ▼                 ▼
```

MediaPipe flips the direction *internally*:

```
Python (webcam demo)
  │
  ├─ RustMediapipeTracker(BaseTracker)  ← rust_bridge.py adapter
  │    │
  │    └─ _skellytracker_rust.MediaPipeTracker(detector_obj, annotator_obj)
  │         │
  │         └─ [Rust] MediaPipeTracker struct
  │              ├─ detector: Py<PyAny>   ← wraps Python MediapipeCompositeDetector
  │              └─ annotator: Py<PyAny>  ← wraps Python MediapipeCompositeAnnotator
  │                   │
  │     detect():      │  Mat → numpy → call Python detector.detect() → extract PointCloud
  │     draw_markers():│  Mat → numpy → call Python annotator.annotate_image() → copy back
  │                   │
  │                   ▼
  │              Python mediapipe (same interpreter, PyO3 call_method1)
```

**The outer flow is still Python→Rust** (webcam demo creates the pyclass). But internally, the Rust tracker calls back into Python for MediaPipe inference.

## Detection pipeline

```
BGR Mat (Rust)
  │
  ├─ 1. Mat → numpy uint8 (H, W, 3) via ndarray → into_pyarray
  │
  ├─ 2. detector.call_method1(py, "detect", (frame_number, numpy_image))
  │      → returns Python MediapipeCompositeObservation
  │
  ├─ 3. Extract from Python obs:
  │      obs.points.names      → Vec<String>
  │      obs.points.xyz        → numpy → Array2<f64> (211, 3)
  │      obs.points.visibility → numpy → Array1<f64> (211,)
  │      obs.has_pose          → bool
  │      obs.has_right_hand    → bool
  │      obs.has_left_hand     → bool
  │      obs.has_face          → bool
  │
  ├─ 4. Build Rust MediaPipeObservation { points: PointCloud, has_pose, ... }
  │
  └─ 5. Stash Python obs as Py<PyAny> for annotate_image to use
```

## Annotation pipeline

Annotation delegates entirely to the Python annotator:

```
annotate_image(Mat)
  │
  ├─ 1. Mat → numpy uint8 (H, W, 3)
  │
  ├─ 2. annotator.call_method1(py, "annotate_image", (numpy_image, last_python_obs))
  │      → returns annotated numpy array
  │
  └─ 3. numpy → copy back into output Mat row-by-row
```

This means **two extra memcpy's per frame** compared to a native Rust implementation (one Mat→numpy for input, one numpy→Mat for output). At 720p this is ~5.5 MB of copies — negligible compared to MediaPipe inference (~10-15ms CPU).

## Data model (Rule #0 applied)

```rust
pub struct MediaPipeObservation {
    pub tracker_type: &'static str,   // "mediapipe_composite"
    pub frame_number: u64,
    pub image_size: (u32, u32),
    pub points: PointCloud,           // 211 points in holistic YAML order
    pub has_pose: bool,
    pub has_right_hand: bool,
    pub has_left_hand: bool,
    pub has_face: bool,
}
```

**211-point composition** (from `mediapipe_holistic.yaml`):

| Slice | Range | Source |
|-------|-------|--------|
| Body | [0:33] | 33 pose landmarks |
| Right hand | [33:54] | 21 hand landmarks, `right_hand_` prefix |
| Left hand | [54:75] | 21 hand landmarks, `left_hand_` prefix |
| Face contour | [75:211] | 136 face mesh contour subset |

**Deferred fields** (not in Phase 1 Rust observation, still available via Python sub-observations):
- `body_world_landmarks` (world-space coords from PoseLandmarker)
- `segmentation_mask` (pose segmentation output)
- `face_blendshapes` (facial expression scores)
- `left_hand_roi` / `right_hand_roi` / `face_roi` (ROI boxes)

## Why this approach

**There are no production-ready Rust bindings for MediaPipe.** Two crates exist (`WasmEdge/mediapipe-rs` — Wasm-only, `ux-mediapipe` — dead at 0.1.0). MediaPipe uses Google's proprietary `.task` format (TFLite-based) — there's no ONNX export path for the full landmarkers. The Python `mediapipe` library wraps Google's highly optimized C++ TFLite runtime and is already fast (~10-15ms CPU).

The reverse PyO3 bridge is pragmatic: the Rust trait system provides the consistent interface freemocap needs, while Python handles the ML inference that has no viable Rust path. In Phase 2, ROI computation + fusion + threading move into Rust for finer control and reduced Python round-trips.

## Key design decision: why no Rust fusion

Phase 1 does NOT replicate the Python fusion logic (face→body, hand→body splicing) in Rust. The Python `MediapipeCompositeObservation.build()` does all fusion internally. Rust simply calls `detector.detect()` and extracts the already-fused PointCloud.

This keeps Phase 1 simple and eliminates a class of bugs (fusion index mismatches, ROI math errors). Phase 2 will port fusion into Rust with native lookup tables.

## Files created

```
skellytracker-rust/src/trackers/mediapipe/
├── mod.rs              # MediaPipeTracker struct + Tracker impl + draw_markers_into
└── observation.rs      # MediaPipeObservation (211 points, detection flags)

skellytracker/trackers/mediapipe_tracker/
└── rust_bridge.py      # RustMediapipeTracker(BaseTracker) adapter + factory
```

## Files modified

```
skellytracker-rust/src/trackers/mod.rs          # + pub mod mediapipe;
skellytracker-rust/src/pyo3_bridge/mod.rs       # + PyMediaPipeTracker (Mutex-wrapped)
skellytracker/io/demo_viewers/webcam_demo_viewer.py  # m-key + r(p)-key toggle
```

## New PyO3 patterns established

1. **Reverse bridge** — `#[new]` accepts `Py<PyAny>` arguments (Python objects passed from the adapter). These are stored as struct fields and called via `.call_method1(py, "method", (...))`.

2. **Py<PyAny> storage** — `Py<PyAny>` is `Send + Sync`, so the tracker struct doesn't need `Mutex` wrapping for these fields. The `last_python_obs: Mutex<Option<Py<PyAny>>>` uses interior mutability only for state changes.

3. **Mat ↔ numpy round-trip** — `mat_to_numpy()` extracts raw pixel data from OpenCV Mat via raw pointer + `copy_from_slice`, wraps as ndarray, and converts to Python numpy via `into_pyarray`. `numpy_to_mat_mut()` reverses the process.

4. **ndarray dynamic→fixed dimensionality** — `PyReadonlyArrayDyn` extracts as `ArrayViewD`, then `.to_owned().into_dimensionality::<Ix2>()` converts to fixed `Array2<f64>`.

## What's deferred to Phase 2

- **Rust-orchestrated pipeline** — Rust holds individual `Py<PyAny>` refs to PoseLandmarker, HandLandmarker, FaceLandmarker instead of one black-box composite detector
- **ROI computation in Rust** — hand/face crop regions computed from pose landmarks with EMA smoothing, arm-direction vectors, last-known-size fallback
- **Rayon parallelism** — `py.allow_threads()` releases GIL, rayon threads re-acquire it for parallel hand+face detection
- **Fusion in Rust** — static lookup tables from `composite_tracker_mappings.py` ported as const arrays
- **`Tracker` trait usable without Python** — `process_image(&mut self, frame_number, &Mat)` works directly (currently requires `Python<'py>` token, so trait methods are stubs)
- **World landmarks + segmentation + blendshapes** — extract additional outputs from Python sub-observations
- **Native TFLite inference** (Phase 3) — convert `.task` → `.tflite`, use `tract` for pure-Rust inference, remove Python dependency

## Verification

- `cargo check` compiles clean
- `poe rebuild` — maturin builds + installs
- `RustMediapipeTracker.create()` — MediaPipe models load (pose + hand + face)
- `process_image()` — returns dict with 211 points, all detection flags
- `annotate_image()` — delegates to Python annotator, returns correctly-shaped numpy
- Factory toggle — `USE_RUST_BACKEND` switches between Rust/Python backends
- Webcam demo — `m` switches to MediaPipe, `p` toggles Rust/Python
