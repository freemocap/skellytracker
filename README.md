# skellytracker

The tracking backend for freemocap. Collects different pose estimation tools behind a consistent API built around a **Tracker → Session → Detector** pipeline. Supports inference on images, webcams, and videos.

## Installation

The base package includes OpenCV-based trackers (Aruco, Charuco) with no extras needed:

```bash
pip install skellytracker

# Or with uv
uv add skellytracker
```

Add extras to enable additional tracker backends. Pick the bundle that matches your hardware:

| Extra | Hardware | What's included |
|---|---|---|
| `recommended-cpu` | Windows/Linux CPU or Apple Silicon | MediaPipe + RTMPose + YOLOX on CPU; enables CoreML EP on macOS |
| `recommended-cuda` | NVIDIA GPU (Linux / Windows) | MediaPipe + RTMPose + YOLOX with TensorRT (best performance) |
| `all-cpu` | Same as `recommended-cpu` | — |
| `all-cuda` | NVIDIA GPU, CUDA 12 only | MediaPipe + RTMPose + YOLOX via CUDA EP (no TensorRT) |
| `all-trt` | NVIDIA GPU, CUDA 12 + TensorRT | Same as `recommended-cuda` |
| `all-directml` | Any GPU on Windows | MediaPipe + RTMPose + YOLOX via DirectML |

```bash
# CPU / Mac
pip install "skellytracker[recommended-cpu]"
uv add "skellytracker[recommended-cpu]"

# NVIDIA GPU
pip install "skellytracker[recommended-cuda]"
uv add "skellytracker[recommended-cuda]"

# Windows with non-NVIDIA GPU
pip install "skellytracker[all-directml]"
uv add "skellytracker[all-directml]"
```

You can also mix extras for more granular control:

```bash
# RTMPose + YOLOX only (no MediaPipe), CUDA backend
pip install "skellytracker[onnx,onnx-cuda]"
uv add "skellytracker[onnx,onnx-cuda]"

# MediaPipe only
pip install "skellytracker[mediapipe]"
uv add "skellytracker[mediapipe]"
```

Available granular extras: `mediapipe`, `onnx`, `onnx-cpu`, `onnx-cuda`, `onnx-trt`, `onnx-directml`.

> **Note:** `onnx-*` backend extras are mutually exclusive — install exactly one.

### Experimental Intel Mac CPU support

For Intel macOS, select `onnx,onnx-cpu` instead of the MediaPipe-containing
`all-cpu` / `recommended-cpu` bundles. The CPU extra pins ONNX Runtime to 1.23.2
on Intel macOS only; its [Intel wheels](https://pypi.org/project/onnxruntime/1.23.2/#files)
require macOS 13 or later. Other platforms retain their existing runtime selection.
The minimum OS version of the complete dependency set has not been validated.

From this checkout, using an isolated Python 3.11 environment:

```bash
uv pip install --python .venv/bin/python ".[onnx,onnx-cpu]" "beartype>=0.22.9,<0.23" pytest
```

The explicit Beartype constraint matches FreeMoCap's existing compatibility
constraint. This is a package-install experiment, not validation of `uv sync`:
the repository lockfile has not been refreshed, and the default development
group also includes SkellyForge. No sibling repositories are needed for this
ONNX-only installation.

On 2026-10-18, macOS 14.3.1 x86_64 / Python 3.11.2 passed 76 focused tests:
10 CPU session/threading and shared-session checks, plus 66 YOLOX/RTMPose image
and video checks. The latter used explicit `CPUExecutionProvider`, the published
reference image, and the existing `freemocap_test_data` synchronized recording
(15 RTMPose frames and 20 YOLOX frames from the first camera). No tests skipped.
Installed versions included ONNX Runtime 1.23.2, ONNX 1.23.2, NumPy 2.4.6,
OpenCV-contrib 5.0.0.93, and Beartype 0.22.9.

Use `execution_provider="cpu"` in `OnnxSessionConfig` to reproduce that provider
selection. This runtime also advertises CoreML on Intel, which automatic
selection can prefer; CoreML was not tested. These results do not establish
realtime throughput, MediaPipe support, or FreeMoCap installer compatibility.

## Quick start

### RTMPose webcam demo

From an environment with `onnx,onnx-cpu` installed:

```bash
.venv/bin/python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.run_demo --provider cpu --camera 0
```

The demo runs RTMPose on every processed frame, updating its crop from keypoints.
YOLOX person detection runs initially, at most 100 processed frames after its
last run, or on the next frame after tracking is lost. Failed detections keep
requesting detection until tracking recovers. This uses the shared `BBoxPolicy`;
the webcam loop contains no separate recovery logic.

Use `--redetect-frames 1` to compare with every-frame person detection, or another
positive interval to tune the tradeoff. The interval counts processed frames,
not camera frames or elapsed seconds. `--no-detector` still skips YOLOX entirely.
The `rtmpose.wholebody.run_demo` entry point uses the same command-line options.

### MediaPipe (pose, hands, face)

```python
import cv2
from skellytracker.core import Tracker, TrackerConfig, DetectionStageConfig, TrackerState
from skellytracker.core.detectors.keypoint_detectors.mediapipe import (
    MediaPipeSession, MediaPipeSessionConfig,
    MediapipePoseDetectorConfig,
    MediapipeHandDetectorConfig,
    MediapipeFaceDetectorConfig,
)

session = MediaPipeSession.create(MediaPipeSessionConfig())
config = TrackerConfig(stages=[
    DetectionStageConfig(name="body",  keypoint_detectors=[MediapipePoseDetectorConfig()]),
    DetectionStageConfig(name="hands", keypoint_detectors=[MediapipeHandDetectorConfig()]),
    DetectionStageConfig(name="face",  keypoint_detectors=[MediapipeFaceDetectorConfig()]),
])
tracker = Tracker.create(config, sessions={"mediapipe": session})
state = TrackerState()

image = cv2.imread("image.jpg")
observation, state = tracker.process_image(image, frame_number=0, state=state)

body_keypoints = observation.stages["body"].keypoints   # Keypoints with .xyz (N,3), .names, .visibility
tracker.close()
```

### RTMPose + YOLOX (ONNX-based)

```python
import cv2
from skellytracker.core import Tracker, TrackerConfig, DetectionStageConfig, TrackerState
from skellytracker.core.detectors.object_detectors.yolox import YoloxPersonDetectorConfig
from skellytracker.core.detectors.keypoint_detectors.rtmpose import RTMPoseDetectorConfig
from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
from skellytracker.core.detectors.object_detectors.yolox import YoloxPersonDetector
from skellytracker.core.detectors.keypoint_detectors.rtmpose import RTMPoseKeypointDetector

session = OnnxSession.create(OnnxSessionConfig(
    batch_size=1,
    models=[
        YoloxPersonDetector.model_spec("yolox-m"),
        RTMPoseKeypointDetector.model_spec("rtmw-x-l_256x192"),
    ],
))
config = TrackerConfig(stages=[
    DetectionStageConfig(
        name="body",
        object_detector=YoloxPersonDetectorConfig(),
        keypoint_detectors=[RTMPoseDetectorConfig()],
    )
])
tracker = Tracker.create(config, sessions={"onnx": session})
state = TrackerState()

image = cv2.imread("image.jpg")
observation, state = tracker.process_image(image, frame_number=0, state=state)

keypoints = observation.stages["body"].keypoints  # 133 whole-body keypoints
tracker.close()
```

### Webcam demo

`skellytracker` installs a console script, so once it's on your environment you can run demos with `uv run skellytracker` (or plain `skellytracker` / `python -m skellytracker` if you've activated the venv yourself).

```bash
# Default demo (MediaPipe holistic: pose + hands + face)
uv run skellytracker

# Pick a different tracker
uv run skellytracker --tracker rtmpose
uv run skellytracker --tracker aruco
uv run skellytracker --tracker charuco

# Use a non-default camera
uv run skellytracker --tracker mediapipe --camera 1

# List available trackers
uv run skellytracker --list
```

`--tracker` chooses among `mediapipe`, `rtmpose`, `aruco`, and `charuco`. Each demo opens a live OpenCV window; press `q` (or close the window) to quit.

For tracker-specific options beyond `--camera` (model choice, execution provider, which body parts to detect, board geometry, etc.), run that tracker's `run_demo` module directly with `--help`:

```bash
# MediaPipe: toggle pose/hands/face
uv run python -m skellytracker.core.detectors.keypoint_detectors.mediapipe.run_demo --no-hands --no-face

# RTMPose: choose model/provider, or skip the YOLOX person detector
uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.run_demo --model rtmw-x-l_384x288 --provider cpu
uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.run_demo --no-detector

# Aruco: choose which marker IDs to look for
uv run python -m skellytracker.core.detectors.keypoint_detectors.aruco.run_demo --ids 0 1 2 3

# Charuco: set board geometry
uv run python -m skellytracker.core.detectors.keypoint_detectors.charuco.run_demo --squares-x 7 --squares-y 5 --square-mm 58
```

## API overview

- **`Tracker`** — top-level orchestrator. Built from a `TrackerConfig` and a `sessions` dict via `Tracker.create()`. Call `process_image(image, frame_number, state)` per frame; it returns `(Observation, TrackerState)`. Call `close()` when done.
- **`Session`** — manages computational resources (model weights, device context) shared across detectors. One session per backend (e.g. one `MediaPipeSession`, one `OnnxSession`).
- **`DetectionStage`** — one stage in the pipeline, configured via `DetectionStageConfig`. Each stage has an optional object detector (to crop person bounding boxes) and one or more keypoint detectors.
- **`Observation`** — per-frame result. `observation.stages["name"].keypoints` returns a `Keypoints` object with `.xyz` `(N, 3)`, `.names`, and `.visibility` arrays.
- **`TrackerState`** — temporal state (bounding box history, smoothing buffers) passed into and returned from each `process_image` call.

## Extending the API

Implement a new keypoint detector by subclassing `KeypointDetector` and registering it:

```python
from skellytracker.core.detectors import KeypointDetector, KEYPOINT_DETECTOR_REGISTRY

class MyDetector(KeypointDetector):
    ...

KEYPOINT_DETECTOR_REGISTRY["my_detector"] = MyDetector
```

## Contributing

Pull requests are welcome! We use [Github Flow](https://docs.github.com/en/get-started/quickstart/github-flow).

1. Fork the repo and create your branch from `main`.
2. Install with dev dependencies: `uv sync --extra recommended-cpu`
3. If you've added a tracker, add tests under `skellytracker/tests/`.
4. Ensure the test suite passes: `pytest skellytracker/tests/` (or `pytest -m 'not video'` to skip slow video tests).
5. Lint: `ruff check skellytracker/`
6. Open a pull request.

---

# GPU setup

For help configuring your GPU for ONNX-based trackers (RTMPose, YOLOX), see the [GPU_SETUP_GUIDE](GPU_SETUP_GUIDE.md).
