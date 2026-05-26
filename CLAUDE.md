# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL — Build Environment Rules

- **NEVER create a `.venv` inside any subdirectory** (e.g. `skellytracker-rust/.venv`). There must be exactly ONE `.venv` at the repo root. A nested `.venv` causes `poe rebuild` to install wheels into the wrong environment, producing stale `.pyd` files and DLL import failures that are extremely confusing to debug. If you see a nested `.venv`, delete it immediately.
- **NEVER use `import *`** (wildcard import) anywhere, in any file, for any reason. Always use explicit named imports. Wildcard imports obscure what symbols are available, break static analysis, and make debugging import failures nearly impossible.
- **Use `poe rebuild` for ALL builds.** No manual `pip install`, no `maturin build` by hand. The single supported build command is `uv run poe rebuild`.
- **Use `logging` for all diagnostic output.** Never `print()` for debug/info messages. Use `logger.debug()` for low-level tracing, `logger.info()` for high-level progress, `logger.error()` for failures. The project already has `logging` set up everywhere — use it.

## Commands

```bash
# Install with dev dependencies and recommended GPU extras
uv sync --extra recommended

# Rebuild Rust native module after changes to skellytracker-rust/
uv run poe rebuild

# Run Rust type check (fast, no Python)
cd skellytracker-rust && cargo check

# Run all tests
pytest skellytracker/tests

# Lint
ruff check skellytracker/

# Run the webcam demo (press h for controls, p toggles Rust/Python)
python skellytracker/webcam_demo.py

# Run the webcam demo with debug logging
python -c "import logging; logging.basicConfig(level=logging.DEBUG); from skellytracker.webcam_demo import *"
```

## Architecture

Skellytracker is the pose-estimation backend for freemocap. It collects different pose estimation tools behind a consistent API built around the **Tracker → Detector / Annotator / Recorder** pattern.

### Core pipeline

1. **`BaseTracker`** (`trackers/base_tracker/base_tracker_abcs.py`) — top-level orchestrator. Composes three sub-objects:
   - **`BaseDetector`** — runs inference on an image, returns a `BaseObservation`
   - **`BaseImageAnnotator`** — draws landmarks/connections onto an image
   - **`BaseRecorder`** — collects observations across frames, can serialize to `.npy` or JSON
2. **`process_image(frame_number, image)`** calls `detector.detect()` and appends to `recorder`. `annotate_image()` is called separately so the caller controls when annotation happens.
3. **`BaseTracker.demo()`** opens a webcam viewer. **`process_video()`** (`io/process_videos/process_single_video.py`) runs frame-by-frame on a video file.

### Canonical data types

- **`PointCloud`** (`trackers/base_tracker/point_cloud.py`) — the canonical data primitive for tracked landmarks. A struct holding names (`tuple[str, ...]`), xyz coordinates (`(N, 3)`), and visibility scores (`(N,)`). Names and coordinates are structurally coupled — the i‑th name always corresponds to the i‑th row. Used throughout: detection → triangulation → filtering → visualization.
- **`BaseObservation`** — abstract base for per-frame results. Every observation carries a `PointCloud` as its canonical data. Subclasses add tracker-specific extras (e.g., raw keypoints for RTMPose). Concrete methods (`to_2d_array()`, `to_tracked_points()`) delegate to the PointCloud.
- **`TrackedObjectDefinition`** (`trackers/base_tracker/tracked_object_definition.py`) — a Pydantic model loaded from YAML that defines the schema of named points and skeleton connections a tracker produces. Supports composition: a YAML with `composed_of` can merge multiple sub-definitions with name prefixes (e.g., `mediapipe_holistic.yaml` composes body + left_hand + right_hand + face_contour).

### YAML-driven tracker schema

Each tracker has a `tracked_object_definitions/` or `names_and_connections/` directory containing YAML files that define `tracked_points` (ordered list of point names) and `connections` (pairs of names forming skeleton edges). This is the single source of truth for point identity and ordering. Detectors use these YAMLs to construct `PointCloud`s with consistent naming; annotators resolve connection name-pairs to array indices for drawing.

### Tracker implementations

#### Python trackers (in `skellytracker/trackers/`)

| Tracker | Location | Notes |
|---------|----------|-------|
| CompositeGPU | `trackers/composite_gpu_tracker/` | RTMO body + RTMPose hands + RTMPose face, single CUDA context with batched inference |
| MediapipeComposite | `trackers/mediapipe_tracker/` | Holistic full-body via MediaPipe Python API (PoseLandmarker + HandLandmarker + FaceLandmarker) |
| RTMPose | `trackers/rtmpose_tracker/` | 133-keypoint whole-body via ONNX Runtime (Python path; has Rust CUDA backend) |
| Charuco | `trackers/charuco_tracker/` | OpenCV Charuco board detection (has Rust backend) |
| BrightestPoint | `trackers/brightest_point_tracker/` | Simple brightest-point tracker (has Rust backend) |
| Legacy (v1) | `trackers/v1/` | Old implementations (YOLO, OpenPose, MMPose, etc.). Not actively maintained. |

#### Rust trackers (in `skellytracker-rust/src/trackers/`)

Each implements the `Tracker` trait and has a `PyO3` bridge pyclass in `skellytracker-rust/src/pyo3_bridge/mod.rs`. Python adapters live in each tracker's `rust_bridge.py`.

| Tracker | Directory | Detection backend | Key pattern |
|---------|-----------|-------------------|-------------|
| BrightestPoint | `brightest_point/` | OpenCV `findContours` | Pure Rust, concrete pyclass |
| Charuco | `charuco/` | OpenCV `detectMarkers`→`detectBoard` | Mutex-wrapped (raw C++ ptrs) |
| RTMPose | `rtmpose/` | ONNX Runtime CUDA (YOLOX + RTMPose) | `ort` crate, CUDA EP, `Provider` enum |
| MediaPipe | `mediapipe/` | Python `mediapipe` via PyO3 reverse bridge | `Py<PyAny>` refs, `call_method1` |

### GPU / ONNX Runtime

GPU trackers use ONNX Runtime with the `ort` crate v2.0.0-rc.12 (`load-dynamic` feature — loads `onnxruntime.dll` at runtime from the pip-installed package).

**RTMPose** (`RtmPoseTracker::new(mode, provider)`):
- Default provider: `CUDA` (~25ms/frame, 1.8× faster than Python CUDA)
- `TensorRT`: wired in but hangs (`ort` crate TRT EP compatibility). Python TRT works (~24ms) but only 1ms faster — not worth debugging.
- `CPU`: fallback (~500-700ms)
- YOLOX detection always uses CUDA (NMS-baked ONNX graph hangs TRT engine compilation)

**CompositeGPU** (Python-only, Rust port planned):
- `CompositeGPUSessionConfig(execution_provider="cuda")`
- Sub-model presets via `SubModelSpec` (light/medium/heavy)

**NVIDIA DLL discovery:** `rust_bridge.py` pre-loads CUDA/cuDNN/ORT DLLs before importing `_skellytracker_rust`, avoiding PATH-order issues on Windows.

The `pyproject.toml` extras (`rtmpose-gpu`, `recommended`) pull in `onnxruntime-gpu` and NVIDIA runtime packages.

### Rust re-architecture (`skellytracker-rust/`)

The Rust crate provides native-performance tracker implementations behind a consistent `Tracker` trait, exposed to Python via PyO3 as `_skellytracker_rust`.

**Core traits** (`src/traits.rs`):
- `Observation` — per-frame detection result: `frame_number()`, `point_cloud()`, `to_json()`
- `Tracker` — orchestrator: `process_image(&mut self, frame, &Mat) → Box<dyn Observation>`, `annotate_image(&self, image, obs) → Mat`
- `PointCloud` (`src/point_cloud.rs`) — the canonical data primitive: names `Vec<String>`, xyz `Array2<f64>`, visibility `Array1<f64>`

**PyO3 bridge** (`src/pyo3_bridge/mod.rs`):
- Each tracker gets a pyclass (e.g. `PyRtmPoseTracker`) wrapping the Rust struct
- `#[new]` receives config params (and optionally Python objects for reverse bridges)
- `process_image(numpy_image) → dict` — runs detection, stashes Rust observation, returns JSON
- `annotate_image(numpy_image, dict) → numpy` — draws from stored observation (NOT JSON-reconstructed)

**Hot-swappable pattern:**
1. `rust_bridge.py` — `Rust*Tracker(BaseTracker)` adapter with `USE_RUST_BACKEND` flag
2. Factory function (`get_*_tracker()`) dispatches to Rust or Python backend
3. Webcam demo: tracker hotkeys + `p` toggles Rust↔Python
4. beartype requires adapters to subclass `BaseTracker` (duck-typing alone fails)

### Key structural conventions

- **`beartype`** runtime type checking is enabled package-wide in `__init__.py` via `beartype_this_package()`.
- **Pydantic for configs**: all detector/tracker configs are Pydantic `BaseModel` subclasses. Detectors and trackers themselves are `@dataclass` classes.
- **YAML for point schemas**: tracked point names and skeleton connections live in YAML under each tracker's `names_and_connections/` directory, loaded by `TrackedObjectDefinition.from_yaml()`.
- **`__main__.py`** references an undefined `main()` function — this path is not currently functional.

### Tests

Tests use `pytest`. `conftest.py` downloads test images from Figshare at session start and provides them as fixtures (`test_image`, `charuco_test_image`). Tests instantiate tracker objects and run `process_image()` on these fixtures, then assert on observation structure and array shapes/values.
