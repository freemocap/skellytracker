# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies (CPU-only)
uv sync

# Install with dev dependencies and recommended GPU extras (CUDA 12 + TensorRT)
uv sync --extra recommended

# Run all tests
pytest skellytracker/tests

# Run a single test file
pytest skellytracker/tests/test_mediapipe_holistic_tracker.py

# Run tests matching a keyword
pytest skellytracker/tests -k "charuco"

# Lint
ruff check skellytracker/

# Format
black skellytracker/
isort skellytracker/

# Run the webcam demo (defaults to mediapipe_holistic)
python -m skellytracker
# Or with a specific tracker:
python -m skellytracker composite_gpu
python -m skellytracker mediapipe_holistic

# Run a specific tracker's demo directly
python -m skellytracker.trackers.mediapipe_tracker
```

**Note:** The `skellytracker` CLI entry point (`__main__.py:cli_main`) calls an undefined `main()` function and will fail at runtime. Individual tracker demos work when invoked directly via `python -m skellytracker.trackers.<tracker_name>`.

Tests require internet access on first run — `conftest.py` downloads test images from Figshare at session start and caches them for subsequent runs.

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
- **`TrackedObjectDefinition`** (`trackers/base_tracker/tracked_object_definition.py`) — a Pydantic model loaded from YAML that defines the schema of named points and skeleton connections a tracker produces. Supports composition: a YAML with `composed_of` can merge multiple sub-definitions with name prefixes (e.g., `rtmo_hybrid.yaml` composes body + right_hand + left_hand + face from separate YAMLs).

### YAML-driven tracker schema

Each tracker has a `names_and_connections/` directory containing YAML files that define `tracked_points` (ordered list of point names) and `connections` (pairs of names forming skeleton edges). This is the single source of truth for point identity and ordering. Detectors use these YAMLs to construct `PointCloud`s with consistent naming; annotators resolve connection name-pairs to array indices for drawing.

### Model Registry

**`utilities/gpu_utils/model_registry.py`** provides a framework-agnostic model resolution and download system:

- **`ModelSource`** — frozen Pydantic model specifying where to get a model (direct URL, Hugging Face repo, or local path).
- **`ModelSpec`** — frozen Pydantic model bundling a `ModelSource` with inference metadata: `format` (onnx/pth/pt/engine), `input_size`, `num_keypoints`, `preprocess_mode`, normalization params, SIMCC split ratio, and batching support. Has convenience constructors for common models (`rtmo_light()`, `rtmpose_hand()`, `mediapipe_hand_landmark()`, etc.).
- **`resolve_model_path(source, cache_dir)`** — downloads and caches models to `~/.cache/skellytracker/models/`. Handles OpenMMLab CDN `.zip` convention (extracts the first `.onnx`) and direct `.onnx` URLs. `resolve_model_paths_parallel()` downloads concurrently via `ThreadPoolExecutor`.
- **`TrackerPreset`** enum — `light`/`medium`/`heavy` performance tiers that bundle model choices for all components.

This centralizes what was previously scattered per-tracker model download logic.

### Detector dispatch and deferred imports

**`trackers/base_tracker/detector_helpers.py`** is the central dispatch mechanism:

- **Factory functions** `create_detector_from_config()` and `create_annotator_from_config()` instantiate the correct detector/annotator class based on config type. Heavy imports (mediapipe, cv2.aruco, onnxruntime) are **deferred to function scope** so they aren't loaded at module import time — critical for multiprocessing scenarios where child processes import this module.
- **Discriminated union types** (`SkeletonDetectorConfig`, `BoardDetectorConfig`, `PointDetectorConfig`) are built dynamically from whichever tracker configs imported successfully. Pydantic's `Discriminator("tracker_type")` routes deserialization to the correct subclass. Each union is built only from members whose optional dependencies are available — a partial install yields a valid (narrower) union rather than crashing.
- Availability flags (`CHARUCO_AVAILABLE`, `MEDIAPIPE_AVAILABLE`, etc.) are set via `try/except ModuleNotFoundError` at module level.

### Tracker implementations

| Tracker | Location | Notes |
|---------|----------|-------|
| CompositeGPU | `trackers/composite_gpu_tracker/` | RTMO body + RTMPose hands + RTMPose face, single CUDA context with batched inference. Configurable via `TrackerPreset` (light/medium/heavy). |
| MediapipeComposite | `trackers/mediapipe_tracker/` | Holistic full-body (pose + hands + face). MediaPipe's native Python API. |
| RTMPose | `trackers/rtmpose_tracker/` | 133-keypoint whole-body via ONNX Runtime (RTMLib). |
| VitPose | `trackers/vitpose_tracker/` | ViT-based pose estimation. |
| Charuco | `trackers/charuco_tracker/` | OpenCV Charuco board detection. |
| BrightestPoint | `trackers/brightest_point_tracker/` | Simple brightest-point-in-frame tracker. |
| Legacy (v1) | `trackers/v1/` | Old tracker implementations (YOLO, OpenPose, MMPose, etc.). Not actively maintained. |

### GPU / ONNX Runtime

All GPU trackers use ONNX Runtime. Shared session utilities live in **`utilities/gpu_utils/ort_session_utils.py`** — used by both `RTMPoseSession` and `CompositeGPUSession` to avoid duplicating provider resolution, session building, CUDA DLL preloading, and batched inference boilerplate.

Execution providers:

- **`cuda`** — CUDA 12 + cuDNN 9. On Windows, skellytracker patches `PATH` and proactively loads NVIDIA DLLs from pip-installed `nvidia-*` packages so users don't need separate CUDA Toolkit/cuDNN system installs.
- **`trt`** — TensorRT engine (2-5x faster than CUDA EP). First run compiles engines (1-5 min); cached thereafter.
- **`directml`** — DirectML for non-NVIDIA GPUs on Windows.
- **`cpu`** — CPU fallback.

The `pyproject.toml` extras (`rtmpose-gpu`, `rtmpose-trt`, `rtmpose-directml`, `recommended`) pull in the correct ONNX Runtime build and NVIDIA runtime packages. Conflicting extras are declared in `[tool.uv].conflicts`. CPU `onnxruntime` is excluded from resolution via `exclude-dependencies`.

### Multi-camera batch processing

**`process_folder_of_videos.py`** processes multiple video files in parallel using `multiprocessing`. Each camera's video is processed in a separate child process, using the detector dispatch system to create tracker instances per-process.

### Key structural conventions

- **`beartype`** runtime type checking is enabled package-wide in `__init__.py` via `beartype_this_package()`.
- **Pydantic for configs**: all detector/tracker configs are Pydantic `BaseModel` subclasses. Detectors and trackers themselves are `@dataclass` classes.
- **YAML for point schemas**: tracked point names and skeleton connections live in YAML under each tracker's `names_and_connections/` directory, loaded by `TrackedObjectDefinition.from_yaml()`.
- **No top-level tracker imports**: all tracker imports in `__init__.py` are commented out. Import directly from each tracker's subpackage (e.g., `from skellytracker.trackers.mediapipe_tracker import MediapipeHolisticTracker`).
- **Lazy loading of heavy dependencies**: `detector_helpers.py` uses deferred imports so that mediapipe, cv2.aruco, and onnxruntime are only imported when actually needed (inside factory functions, not at module level).

### Tests

Tests use `pytest`. `conftest.py` downloads test images from Figshare at session start and provides them as fixtures (`test_image`, `charuco_test_image`). Tests instantiate tracker objects and run `process_image()` on these fixtures, then assert on observation structure and array shapes/values. Most tests are structural/smoke tests — they verify the pipeline runs without error and produces correctly shaped output.