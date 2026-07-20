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

## Session as Batch Coordinator

For multi-camera setups, the `OnnxSession` is also the point where batching happens. `DetectionStage.run_batch()` collects preprocessed tensors from all N cameras and hands them to the session as a single stacked array. The session fires one GPU call and returns N results:

```python
# In DetectionStage.run_batch():
tensors = {cam_id: detector.preprocess(images[cam_id]) for cam_id in images}
stacked = np.stack([tensors[k].tensor for k in ordered_keys])   # (N, 3, H, W)
raw_outputs = session.run_batched(model_name, stacked)           # one GPU call
per_camera = {k: raw_outputs[i] for i, k in enumerate(ordered_keys)}
```

The session itself remains agnostic to pipeline structure — it only stacks inputs and splits outputs. All orchestration (which models to call, in what order, how to route results) lives in `DetectionStage`.

This is why the session is the right place for batching: it already owns the GPU context and the loaded model weights. Batching is purely an inference-layer concern — it changes nothing about preprocessing, temporal processing, or output structure.

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

    def run(self, model_name: str, inputs: dict) -> list:
        """Single-image inference. inputs contains a (1, 3, H, W) tensor."""
        ...

    def run_batched(self, model_name: str, tensor: NDArray) -> NDArray:
        """Multi-camera inference. tensor is (N, 3, H, W); returns (N, ...) raw output."""
        ...
```

`ONNXSessionConfig` lists all the models the session should load (`list[ModelSpec]`) along with the execution provider, device ID, and `batch_size`. Models are loaded with dynamic batch dimensions (via ONNX surgery for providers that support it) so that any `N ≤ batch_size` can be passed at runtime. See [08-onnx-batching-and-coreml.md](./08-onnx-batching-and-coreml.md).

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
