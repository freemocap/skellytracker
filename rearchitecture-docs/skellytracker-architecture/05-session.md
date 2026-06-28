# Session

A `Session` manages the computational resources required to run detectors: GPU memory, loaded model weights, and backend-specific handles. It is created once per backend and shared across all detectors in a `Tracker` that use that backend. Detectors do not own resource lifecycle — the `Session` does.

## Why a Top-Level Session

In the current architecture, ONNX sessions and GPU warmup live inside each detector. This works for single-tracker use but makes it awkward to share a CUDA context across models, control memory limits centrally, or tear down resources cleanly. Moving session ownership to the top level gives one place to handle:

- **Shared context**: all models on the same device and backend share a single session object, avoiding redundant allocations.
- **Device selection**: pick the best available GPU (or CPU fallback) once, not per-model.
- **Coordinated warmup**: run all models through a warmup pass together before the first real frame.
- **Clean teardown**: `session.close()` releases everything in one call.

## Abstract Interface

```python
class Session(ABC):
    @classmethod
    @abstractmethod
    def create(cls, config: SessionConfig) -> "Session": ...

    @abstractmethod
    def close(self) -> None: ...
```

A `Tracker` owns one `Session` per backend it uses. Each detector receives the appropriate session type at construction time via `Tracker.create()`.

## Concrete Session Types

### ONNXSession

Manages all ONNX models used by a `Tracker` — body, hand, face, or any other ONNX-based detector — in a single object. Bundling them together is intentional: models on the same device share a CUDA context, and keeping them in one session avoids redundant context creation and allows coordinated warmup and teardown.

```python
@dataclass
class ONNXSession(Session):
    config: ONNXSessionConfig
    # internals: dict of model_name → ort.InferenceSession

    @classmethod
    def create(cls, config: ONNXSessionConfig) -> "ONNXSession":
        # resolves execution provider, selects device,
        # downloads/loads all model files, warms up
        ...
```

`ONNXSessionConfig` lists all the models the session should load (`list[ModelSpec]`) along with the execution provider and device ID. Detectors reference their model by name when they need to run inference.

**Execution providers:**

| Provider | When to use |
|----------|-------------|
| `cuda` | NVIDIA GPU, CUDA 12 + cuDNN 9 |
| `trt` | NVIDIA GPU, TensorRT (2-5× faster; first run compiles engines) |
| `directml` | Non-NVIDIA GPU on Windows |
| `cpu` | CPU fallback |

Provider resolution logic (probe → fallback chain) lives inside `ONNXSession.create()`, not scattered across detectors.

### MediaPipeSession

Manages MediaPipe task handles and GPU delegate configuration. MediaPipe has its own resource lifecycle that is incompatible with ONNX Runtime, so it gets its own session type.

```python
@dataclass
class MediaPipeSession(Session):
    config: MediaPipeSessionConfig
    # internals: MediaPipe task handles per model
```

### Other Backends

Additional session types follow the same pattern: one concrete `Session` subclass per backend (e.g., `DeepLabCutSession`, `TorchSession`), each owning all models for that backend within a given `Tracker`.

## Wiring to Detectors

A `Tracker` holds one session per backend. When `Tracker.create()` constructs detectors, it passes each detector the session it needs:

```python
sessions = {
    "onnx": ONNXSession.create(onnx_config),
    "mediapipe": MediaPipeSession.create(mp_config),
}
tracker = Tracker.create(tracker_config, sessions=sessions)
```

Each detector config declares which backend it uses. `Tracker.create()` resolves the right session and passes it to the detector at construction time. Detectors do not hold references to the session dict — only to their specific session.

## Session vs TrackerState

- **Session** = static resources (GPU memory, loaded model weights). Never changes after creation.
- **TrackerState** = dynamic per-frame data (smoothed bounding boxes, filter coefficients). Changes every frame.

Session is long-lived and device-bound. TrackerState is frame-by-frame data that travels with the Observation, not with the Session.
