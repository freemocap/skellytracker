# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies and all trackers (CPU)
uv sync --extra all-cpu
# Or for NVIDIA GPU with TensorRT:
uv sync --extra all-trt

# Run all tests
pytest skellytracker/tests

# Run a single test file
pytest skellytracker/tests/test_mediapipe_holistic_tracker.py

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

| Tracker | Location | Notes |
|---------|----------|-------|
| CompositeGPU | `trackers/composite_gpu_tracker/` | RTMO body + RTMPose hands + RTMPose face, single CUDA context with batched inference. Configurable via `SubModelSpec` presets (light/medium/heavy). |
| MediapipeComposite | `trackers/mediapipe_tracker/` | Holistic full-body (pose + hands + face). MediaPipe's native Python API. |
| RTMPose | `trackers/rtmpose_tracker/` | 133-keypoint whole-body via ONNX Runtime (RTMLib). |
| VitPose | `trackers/vitpose_tracker/` | ViT-based pose estimation. |
| Charuco | `trackers/charuco_tracker/` | OpenCV Charuco board detection. |
| BrightestPoint | `trackers/brightest_point_tracker/` | Simple brightest-point-in-frame tracker. |
| Legacy (v1) | `trackers/v1/` | Old tracker implementations (YOLO, OpenPose, MMPose, etc.). Not actively maintained. |

### GPU / ONNX Runtime

All GPU trackers use ONNX Runtime. The execution provider is selected via config:

- **CompositeGPU**: `CompositeGPUSessionConfig(execution_provider="cuda")`. Sub-model selection uses `SubModelSpec` presets — `CompositeGPUSessionConfig.preset("light")` for rtmo-s body, `"medium"` (default) for rtmo-m, `"heavy"` for rtmo-l. Hand and face models can be overridden per-component via `body_spec`/`hand_spec`/`face_spec` fields.
- **RTMPose**: `RTMPoseDetectorConfig.resolved_provider()`.

Execution providers:

- **`cuda`** — CUDA 12 + cuDNN 9. On Windows, skellytracker patches `PATH` and proactively loads NVIDIA DLLs from pip-installed `nvidia-*` packages so users don't need separate CUDA Toolkit/cuDNN system installs.
- **`trt`** — TensorRT engine (2-5x faster than CUDA EP). First run compiles engines (1-5 min); cached thereafter.
- **`directml`** — DirectML for non-NVIDIA GPUs on Windows.
- **`cpu`** — CPU fallback.

The `pyproject.toml` extras (`rtmpose-gpu`, `rtmpose-trt`, `rtmpose-directml`, `recommended`) pull in the correct ONNX Runtime build and NVIDIA runtime packages. Conflicting extras are declared in `[tool.uv].conflicts`. CPU `onnxruntime` is excluded from resolution via `exclude-dependencies`.

### Key structural conventions

- **`beartype`** runtime type checking is enabled package-wide in `__init__.py` via `beartype_this_package()`.
- **Pydantic for configs**: all detector/tracker configs are Pydantic `BaseModel` subclasses. Detectors and trackers themselves are `@dataclass` classes.
- **YAML for point schemas**: tracked point names and skeleton connections live in YAML under each tracker's `names_and_connections/` directory, loaded by `TrackedObjectDefinition.from_yaml()`.
- **`__main__.py`** references an undefined `main()` function — this path is not currently functional.

### Tests

Tests use `pytest`. `conftest.py` downloads test images from Figshare at session start and provides them as fixtures (`test_image`, `charuco_test_image`). Tests instantiate tracker objects and run `process_image()` on these fixtures, then assert on observation structure and array shapes/values.
