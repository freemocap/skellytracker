# skellytracker-rust

Rust re-architecture of the skellytracker pose-estimation backend. Follows the same PyO3 bridge pattern as [skellycam-rust](https://github.com/freemocap/skellycam), built using the [rearchitecture playbook](../rearchitecture-docs/rearchitecture-playbook/README.md) methodology.

## Architecture

```
┌─ Python layer ───────────────────────────────────────────────┐
│  skellytracker.trackers.<tracker>.rust_bridge                 │
│    │                                                          │
│    │ import _skellytracker_rust                                │
│    ▼                                                          │
├─ PyO3 bridge ────────────────────────────────────────────────┤
│  src/pyo3_bridge/mod.rs                                       │
│    #[pymodule] fn _skellytracker_rust                         │
│    #[pyclass] PyBrightestPointTracker  (thin adapter)          │
│    │                                                          │
├─ Pure Rust layer ────────────────────────────────────────────┤
│  traits.rs         Tracker / Detector / Annotator / Observation │
│  point_cloud.rs    Canonical data type (ndarray-backed)        │
│  recorder.rs       Collect observations; serialize to NPY/JSON │
│  trackers/         Tracker implementations                     │
│    brightest_point/   OpenCV contour-based bright patch detect │
│    charuco/           (planned)                                │
│    rtmpose/           (planned)                                │
└──────────────────────────────────────────────────────────────┘
```

### Trait decomposition

Every tracker implements `Tracker`, which composes `process_image` and `annotate_image`:

```rust
pub trait Tracker {
    fn process_image(&mut self, frame_number: u64, image: &Mat) -> Box<dyn Observation>;
    fn annotate_image(&self, image: &Mat, obs: &dyn Observation) -> Mat;
}
```

Trackers that benefit from further decomposition can optionally implement `Detector` and `Annotator` traits separately — the `Tracker` impl delegates to them. This matches the Python `BaseTracker → BaseDetector / BaseImageAnnotator` pattern.

### Canonical data type: `PointCloud`

All observations carry a `PointCloud` — an `ndarray`-backed struct coupling named landmarks to their XYZ coordinates and visibility scores. The name-to-index mapping is structural: the i-th name always corresponds to the i-th row. This replaces the Python `PointCloud` which used numpy arrays with the same invariant.

## Python → Rust architecture mapping

> This follows [Step 3](../rearchitecture-docs/rearchitecture-playbook/03-separate-python-concerns.md) (separate Python-specific from universal) and [Step 4](../rearchitecture-docs/rearchitecture-playbook/04-design-rust-architecture.md) (design from invariants) of the rearchitecture playbook.

### The key insight

Most of the Python architecture's complexity comes from one root cause: **processes have separate address spaces.** Everything — shared memory, PubSub, pickle serialization, DTOs, staggered spawn, heartbeat monitoring — traces back to `multiprocessing.Process`.

In Rust, threads share the heap. This single difference eliminates ~60-70% of the Python architecture's complexity. The remaining 30-40% is the actual problem: running pose detection on frames.

**We are not porting Python to Rust.** We are re-solving the same problem with a different set of constraints. The architecture looks different — that's the point.

### Universal concept translations (from playbook Step 4)

| Universal Concept | Python Implementation | Rust Implementation |
|-------------------|----------------------|---------------------|
| Shared mutable flag | `multiprocessing.Value("b")` | `Arc<AtomicBool>` |
| Shared configuration | `multiprocessing.Value` per field | `Arc<Mutex<Config>>` |
| Command dispatch | PubSub topic + polling | `mpsc::Sender<CommandEnum>` |
| Thread-safe lazy init | Module-level global + `get_or_create_*` | `OnceLock<T>` |
| Graceful shutdown signal | `multiprocessing.Value("b")` + atexit | `Arc<AtomicBool>` + Drop impls |
| Structured binary data | numpy recarray with dtype | `#[repr(C)]` struct + `bytemuck` |

### Python-specific patterns we explicitly do NOT carry over

| Python Pattern | Why It Exists | Why It Doesn't Apply in Rust |
|---------------|---------------|------------------------------|
| `multiprocessing.Process` | CPython GIL serializes threads | `std::thread::spawn` runs in true parallel |
| `multiprocessing.Value` / `.Queue` | Processes can't share heap objects | `Arc<T>`, typed channels — same address space |
| Pickle serialization of all messages | `multiprocessing.Queue` requires it | Channels pass owned typed data, zero serialization |
| `beartype` runtime type checking | Python type hints are not enforced | Compile-time — the type system checks everything |
| `atexit` handlers + try/finally | Python exit is cooperative | `Drop` impls are compiler-guaranteed, even on panic |
| DTO/recreate pattern | Each process maps SHM independently | Threads share the heap — no "recreating" needed |
| Polling PubSub subscriptions | No blocking receive across processes | `recv()` blocks until data arrives |

### skellytracker-specific mappings

These are the concrete Python→Rust translations for the tracker framework:

| Concern | Python | Rust | What changed |
|---------|--------|------|--------------|
| **Tracker framework** | `BaseTracker` abstract class with `BaseDetector` / `BaseImageAnnotator` / `BaseRecorder` composition | `Tracker` trait with optional `Detector` / `Annotator` decomposition | Traits replace abstract classes. No runtime `isinstance()` checks — trait bounds enforced at compile time. |
| **Config validation** | Pydantic `BaseModel` with runtime field validators | `#[derive(Deserialize)]` + `serde` — compile-time struct shape, runtime value parsing | Struct shape checked at compile time. YAML deserialization is still runtime (must be, since YAML is dynamic), but the target type is verified. |
| **Point schema** | YAML → `TrackedObjectDefinition` (Pydantic) → `PointCloud` | YAML → serde struct → `PointCloud` (ndarray-backed) | Same concept, zero-copy via ndarray slices. Names and coordinates are structurally coupled — the compiler can't enforce this, so assertions validate at construction time. |
| **Array data** | `numpy` ndarray — C-contiguous, untyped at the element level | `ndarray::Array2<f64>` — contiguous, type-safe at every index | Same row-major layout. ndarray provides equivalent slicing (`s![.., ..2]`), mapping (`map_axis`), and indexing. No Python overhead for element access. |
| **Observation** | `BaseObservation` dataclass with `.point_cloud` property | `Observation` trait with `fn point_cloud(&self) -> &PointCloud` | Trait object instead of duck-typing. `as_any()` enables downcasting for tracker-specific extensions (like `BrightestPointObservation.patches`). |
| **Recorder** | `BaseRecorder` abstract class, collects observations per frame | `Recorder` struct with `Vec<Box<dyn Observation>>` | Same append-and-serialize pattern. NPY output uses `ndarray-npy`. JSON output uses `serde_json`. No pickle — data is typed through the `Observation` trait. |
| **Image I/O** | `cv2.imread` → numpy, `cv2.imwrite` to disk | `opencv::imgcodecs::imread` → `Mat`, same for write | Same OpenCV calls, different binding layer. The `opencv` crate directly wraps the C++ API. |
| **CV operations** | `cv2.cvtColor`, `cv2.threshold`, `cv2.findContours`, `cv2.moments` | `opencv::imgproc::cvt_color`, `imgproc::threshold`, `imgproc::find_contours`, `imgproc::moments` | Same OpenCV functions, idiomatic Rust wrappers. Return types are `Result<T, Error>` instead of bare values or tuples. |
| **Drawing** | `cv2.drawMarker` with positional args | `imgproc::draw_marker` with typed params | Same OpenCV call. Marker type, size, thickness are explicit named constants (`MARKER_CROSS`, `LINE_8`) rather than magic integers. |
| **Plugin registration** | Dynamic imports + registry dict at module load | Module declarations — `pub mod trackers` tree is the registry | Compile-time module resolution. Adding a tracker means creating a module and declaring it in `mod.rs` and `lib.rs`. No runtime discovery needed. |
| **GPU inference** (planned) | `onnxruntime.InferenceSession` with thread-local sessions | `ort` crate — same ONNX Runtime C API, no GIL contention | Same inference engine. Rust can hold multiple sessions across threads without GIL serialization. Tensor I/O via `ndarray` views instead of numpy copies. |
| **GPU memory** (planned) | Python GC + explicit `del session` to free CUDA allocations | `Drop` impls — deterministic, compiler-guaranteed, runs when the session goes out of scope | No "did we remember to call `del`?" — the compiler ensures cleanup. |

## PyO3 bridge strategy (copied from skellycam)

| Concern | skellycam | skellytracker |
|---------|-----------|---------------|
| Cargo `[lib]` name | `skellycam` | `skellytracker` |
| `#[pymodule]` name | `_skellycam_rust` | `_skellytracker_rust` |
| maturin `module-name` | `_skellycam_rust` | `_skellytracker_rust` |
| `python-source` directory | **none** | **none** |
| Python package wrapping native | No — `.pyd` lives in site-packages directly | Same |
| Hot-swappable adapter | `USE_RUST_BACKEND` flag in `camera_group_manager.py` | `USE_RUST_BACKEND` flag in `rust_bridge.py` |
| DLL bundling | N/A (no OpenCV) | `os.add_dll_directory()` before import |

The `.pyd` is installed directly into site-packages as `_skellytracker_rust.pyd` — no Python package wrapper, no `python/` source directory. On Windows, `os.add_dll_directory("C:/tools/opencv/build/x64/vc16/bin")` is called before import so the OS loader finds the OpenCV DLLs.

## Build & install

```bash
# Prerequisites (Windows):
#   choco install opencv llvm

# Rebuild everything (Rust crate + Python package) — runs maturin with verbose cargo output:
poe rebuild

# Rust-only check (fast, no Python needed):
cd skellytracker-rust
cargo check

# Rust release build (no Python needed):
cd skellytracker-rust
cargo build --release

# Verify the native module is importable:
python -c "import _skellytracker_rust; print(_skellytracker_rust.BrightestPointTracker(3, 200))"
```

## Test

```bash
# Webcam demo (Rust backend):
python skellytracker-rust/test_demo.py

# Webcam demo (Python fallback):
python skellytracker-rust/test_demo.py --python

# Single image test:
python skellytracker-rust/test_demo.py --image path/to/image.jpg

# Run pytest suite:
poe test
```

## Hot-swappable backends

Set `USE_RUST_BACKEND` in `skellytracker/trackers/brightest_point_tracker/rust_bridge.py`:

```python
USE_RUST_BACKEND = True   # Rust PyO3 engine
USE_RUST_BACKEND = False  # Original Python OpenCV engine
```

The factory function `get_brightest_point_tracker()` returns the appropriate backend. Callers don't need to know which one they're talking to — both expose the same interface:

```python
from skellytracker.trackers.brightest_point_tracker.rust_bridge import get_brightest_point_tracker

tracker = get_brightest_point_tracker(num_points=3, luminance_threshold=200)
result = tracker.process_image(0, image)                      # runs detection ONCE
annotated = tracker.annotate_image(image, observation=result)  # draws from result
```

## Source tree

```
skellytracker-rust/
├── Cargo.toml              Crate manifest — deps, lib name, features
├── pyproject.toml          maturin config — module-name, no python-source
├── build.rs                Build script — copies OpenCV DLLs to cargo target dirs
├── .cargo/config.toml      OpenCV env vars (LINK_PATHS, INCLUDE_PATHS)
├── test_demo.py            End-to-end webcam demo (Rust + Python fallback)
└── src/
    ├── lib.rs              Crate root — declares modules
    ├── traits.rs           Tracker, Detector, Annotator, Observation traits
    ├── point_cloud.rs      PointCloud struct (ndarray-backed, serde)
    ├── recorder.rs         Frame-collector, NPY/JSON serialization
    ├── pyo3_bridge/
    │   ├── mod.rs          #[pymodule] _skellytracker_rust, PyBrightestPointTracker
    │   └── types.rs        Python-facing dataclass equivalents (placeholder)
    └── trackers/
        ├── mod.rs
        └── brightest_point/
            ├── mod.rs      BrightestPointTracker — contour-based bright patch detect
            └── observation.rs  BrightestPointObservation, BrightPatch
```

## Trackers

| Tracker | Status | Description |
|---------|--------|-------------|
| `brightest_point` | Done | OpenCV contour-based bright patch detection with centroid extraction |
| `charuco` | Planned | OpenCV Charuco board detection |
| `rtmpose` | Planned | ONNX Runtime whole-body pose estimation (133 keypoints) |
| `mediapipe_holistic` | Planned | MediaPipe holistic (pose + hands + face) |
| `composite_gpu` | Planned | Multi-model GPU pipeline (RTMO body + RTMPose hands/face) |

## Dependencies

| Crate | Purpose |
|-------|---------|
| `opencv 0.98` | Computer vision (thresholding, contours, moments, drawing) |
| `ndarray 0.16` | Multi-dimensional array math (replaces numpy) |
| `ndarray-npy 0.9` | NPY file I/O |
| `pyo3 0.23` | Python bindings |
| `numpy 0.23` | PyO3 ↔ numpy array interop |
| `serde` / `serde_json` | Serialization (PointCloud, observations, recorder output) |
