# Session

A `Session` manages the computational resources required to run detectors: GPU memory, ONNX Runtime inference sessions, MediaPipe handles, downloaded model weights. It is created once per device and shared across all detectors in a `Tracker`. Detectors do not own resource lifecycle — the `Session` does.

## Why a Top-Level Session

In the current architecture, ONNX sessions and GPU warmup live inside each detector. This works for single-tracker use but makes it awkward to share a CUDA context across models, control memory limits centrally, or tear down resources cleanly. Moving session ownership to the top level gives one place to handle:

- **Device selection**: pick the best available GPU (or CPU fallback) once, not per-model.
- **Shared CUDA context**: all models on the same device share a single context, avoiding redundant allocations.
- **Coordinated warmup**: run all models through a warmup pass together before the first real frame.
- **Clean teardown**: `session.close()` releases everything in one call.

## Interface

```python
@dataclass
class Session:
    config: SessionConfig
    # internals: ort.InferenceSession instances, MediaPipe modules, etc.

    @classmethod
    def create(cls, config: SessionConfig) -> "Session":
        # resolves execution provider, selects device, loads/downloads models, warms up
        ...

    def close(self) -> None:
        # releases all GPU/CPU resources
        ...
```

Detectors receive the `Session` at construction time and call methods on it to run inference. They never hold a reference to raw ONNX sessions or other backend handles directly — those remain private to the `Session`.

## SessionConfig

```python
class SessionConfig(BaseModel):
    execution_provider: Literal["cuda", "trt", "directml", "cpu"] = "cuda"
    device_id: int = 0          # which GPU to use (0 = auto-select best)
    models: list[ModelSpec]     # which model files to load (downloaded on demand)
```

`ModelSpec` describes a model by name/version/source; the `Session` resolves the actual file paths and downloads if needed (same as the current `SubModelSpec` / model-download pattern in `CompositeGPUSession`).

## Execution Providers

The same provider options as the current architecture apply:

| Provider | When to use |
|----------|-------------|
| `cuda` | NVIDIA GPU, CUDA 12 + cuDNN 9 |
| `trt` | NVIDIA GPU, TensorRT (2-5× faster; first run compiles engines) |
| `directml` | Non-NVIDIA GPU on Windows |
| `cpu` | CPU fallback |

Provider resolution logic (probe → fallback chain) lives inside `Session.create()`, not scattered across detectors.

## Session vs TrackerState

These two are sometimes confused:

- **Session** = static resources (GPU memory, loaded model weights). Never changes after creation.
- **TrackerState** = dynamic per-frame data (smoothed bounding boxes, filter coefficients). Changes every frame.

Session is long-lived and device-bound. TrackerState is frame-by-frame data that travels with the Observation, not with the Session.
