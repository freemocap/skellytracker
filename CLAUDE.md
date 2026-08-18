# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies + tracker backends (CPU / Apple Silicon)
uv sync --extra recommended-cpu
# Or NVIDIA GPU with TensorRT:
uv sync --extra recommended-cuda

# Run the tests
pytest skellytracker/tests
# Skip the slow video tests
pytest -m 'not video'
# A single test file
pytest skellytracker/tests/test_mediapipe_detectors.py

# Lint / format
ruff check skellytracker/
black skellytracker/
isort skellytracker/

# Run the live webcam demo (defaults to mediapipe)
python -m skellytracker
python -m skellytracker --tracker rtmpose
python -m skellytracker --list

# Run a specific detector's demo directly (more options via --help)
python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.run_demo --help
```

## Architecture

skellytracker is the pose-estimation backend for freemocap. It collects
pose-estimation tools behind one consistent API built around a **Tracker →
Session → Detector** pipeline, implemented in the `skellytracker/core/` package.
`beartype` runtime type-checking is enabled package-wide in `__init__.py` via
`beartype_this_package()`.

### Core pipeline

1. **`Tracker`** (`core/tracker/tracker.py`) — top-level orchestrator. A
   `@dataclass` of `stages` + `sessions`, built via `Tracker.create(config, sessions)`.
   `process_image(image, frame_number, state, timestamp_ms=None)` runs all stages
   and returns `(Observation, TrackerState)`; `process_batch(images, frame_number, states, ...)`
   runs N cameras in one batched call. `close()` releases resources. The tracker
   is stateless between calls — temporal data lives in `TrackerState`.
2. **`DetectionStage`** (`core/tracker/detection_stage.py`) — the compositional
   unit. Binds one optional `ObjectDetector` (crop) and one or more
   `KeypointDetector`s, and can carry child stages that run on its crop
   (hierarchical body → hands/face). `run()` / `run_batch()`.
3. **`ObjectDetector` / `KeypointDetector`** (`core/detectors/`) — the two
   primitive detection units, built from Pydantic configs via
   `KEYPOINT_DETECTOR_REGISTRY` / `OBJECT_DETECTOR_REGISTRY` and
   `build_keypoint_detector` / `build_object_detector`.
4. **`Session`** (`core/sessions/`) — owns backend resources (GPU memory, model
   weights, handles); created once per backend and shared across detectors.
   Concrete: `OnnxSession` (`OnnxSessionConfig(batch_size, models)`),
   `MediaPipeSession`, `CpuSession`.

### Core data types

- **`Keypoints`** (`core/data_primitives/keypoints.py`) — named keypoints:
  `.xyz` `(N, 3)`, `.names` `tuple[str, ...]` (i-th name ↔ i-th row),
  `.visibility` `(N,)`.
- **`BoundingBox`** (`core/data_primitives/bounding_box.py`).
- **`Observation`** (`core/data_primitives/observation.py`) — per-frame result;
  `observation.stages["name"].keypoints` (a `StageObservation` → `Keypoints`).
  This is the stable contract for downstream consumers.
- **`TrackerState`** (`core/tracker/tracker_state.py`) — explicit external
  temporal state, passed into and returned from each call (`StageState`,
  `BBoxSmoothingState`, `KeypointSmoothingState`).
- **`DataStore`** (`core/data_primitives/data_store.py`) — collects and
  serializes observations to `.npy` / JSON.

### Detectors

| Backend | Location | Notes |
|---------|----------|-------|
| MediaPipe | `core/detectors/keypoint_detectors/mediapipe` | pose / hands / face (native MediaPipe API) |
| RTMPose | `core/detectors/keypoint_detectors/rtmpose` | body / face / hand / wholebody (133 keypoints) via ONNX |
| YOLOX | `core/detectors/object_detectors/yolox` | person object detector (crops for top-down) |
| Aruco | `core/detectors/keypoint_detectors/aruco` | OpenCV Aruco marker detection |
| Charuco | `core/detectors/keypoint_detectors/charuco` | OpenCV Charuco board detection |
| Precomputed | `core/detectors/object_detectors/precomputed` | supply externally computed bounding boxes |

Each keypoint detector defines its keypoint names and skeleton connections (see
`core/detectors/keypoint_detectors/_schema_loader.py`); annotators
(`core/annotation/`) resolve connection name-pairs to array indices for drawing.

### GPU / ONNX Runtime

The ONNX-backed detectors (RTMPose, YOLOX) run through ONNX Runtime; the
execution provider is selected per session:

- **`cuda`** — CUDA 12 + cuDNN 9. On Windows, skellytracker preloads the
  pip-installed `nvidia-*` DLLs so no separate CUDA Toolkit / cuDNN system
  install is required.
- **`trt`** — TensorRT (fastest). First run compiles engines (1–5 min); cached thereafter.
- **`directml`** — non-NVIDIA GPUs on Windows.
- **`cpu`** — fallback; enables the CoreML EP on Apple Silicon.

`pyproject.toml` extras (`onnx-cuda`, `onnx-trt`, `onnx-directml`, `onnx-cpu`, and
the `recommended-*` / `all-*` bundles) pull in the matching ONNX Runtime + NVIDIA
runtime packages. The `onnx-*` backend extras are mutually exclusive — install
exactly one.

### Temporal processing

`core/temporal_processing/` holds the cross-frame smoothing: bounding-box policy
and smoothing (`bbox_policy.py`, `bbox_smoothing.py`), keypoint filtering
(`keypoint_filtering.py`), and the keypoint reset policy
(`keypoint_reset_policy.py`). All persisted state lives in `TrackerState`.

### IO & demos

`core/io/`: `process_video` (one file), `process_video_list` / `process_folder`
(multiple synchronized videos, batched via `Tracker.process_batch`), `DemoManager`
(live webcam viewer, used by each detector's `run_demo.py`), `ProcessingTimer`,
`TrackerMapping`. The `skellytracker` / `python -m skellytracker` CLI
(`__main__.py:cli_main`) runs the webcam demos (`--tracker`, `--camera`,
`--rotate`, `--list`).

### Tests

Tests use `pytest` under `skellytracker/tests/`. `conftest.py` provides shared
test-image fixtures. Tests build trackers and run `process_image` / `process_batch`
on the fixtures, then assert on observation structure and array shapes. Slow
tests are marked `video` (skip with `pytest -m 'not video'`).
```
