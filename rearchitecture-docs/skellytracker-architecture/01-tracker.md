# Tracker

The `Tracker` is the top-level pipeline object. It is the only public entry point for running pose estimation: callers pass in a frame and the current `TrackerState`, and receive an `Observation` and an updated state. Everything else — detection stages, sessions, smoothing — is an implementation detail behind this interface.

A `Tracker` owns a list of `DetectionStage`s. Stages are ordered; each runs in sequence on the input image (or a crop derived from a parent stage). Stages can be nested, so the full pipeline is a tree rather than a flat list.

The `Tracker` does not own mutable state itself. It reads `TrackerState` on each call and returns an updated copy.

## Interface

```python
@dataclass
class Tracker:
    stages: list[DetectionStage]
    sessions: dict[str, Session]

    def process_image(
        self,
        image: NDArray[np.uint8],
        frame_number: int,
        state: TrackerState,
        timestamp_ms: int | None = None,
    ) -> tuple[Observation, TrackerState]:
        ...

    def process_batch(
        self,
        images: dict[str, NDArray[np.uint8]],
        frame_number: int,
        states: dict[str, TrackerState],
        timestamp_ms: int | None = None,
    ) -> tuple[dict[str, Observation], dict[str, TrackerState]]:
        ...

    @classmethod
    def create(cls, config: TrackerConfig, sessions: dict[str, Session]) -> "Tracker":
        ...
```

`process_image` is the single-camera hot path. It runs all stages, merges their outputs into an `Observation`, and returns the updated `TrackerState`.

`process_batch` is the multi-camera entry point. Images and states are keyed by camera ID (a caller-chosen string, e.g. `"cam0"`). Internally it delegates to `DetectionStage.run_batch()`, which performs true batched GPU inference across all cameras in a single session call per model. Each camera's temporal state is updated independently. See [10-multi-camera-batching.md](./10-multi-camera-batching.md) for full detail.

`process_image` is a convenience wrapper around `process_batch` with a single-element dict — it does not have a separate implementation.

## Configuration

A `TrackerConfig` (Pydantic model) describes the full stage tree declaratively. This is the primary way to build a `Tracker`: define the config in YAML or Python, call `Tracker.create()`, and receive a fully wired tracker ready to run.

```python
config = TrackerConfig(
    stages=[
        DetectionStageConfig(
            object_detector=YOLODetectorConfig(...),
            keypoint_detectors=[RTMPoseDetectorConfig(...)],
            children=[
                DetectionStageConfig(
                    object_detector=FaceDetectorConfig(...),
                    keypoint_detectors=[FaceKeypointDetectorConfig(...)],
                    additional_config=...
                ),
            ],
            additional_config=...
        ),
    ]
)
tracker = Tracker.create(config, session=session)
```

## Lifecycle

### Single-camera

1. `session = Session.create(config)` — allocate GPU/CPU resources
2. `tracker = Tracker.create(tracker_config, sessions={"onnx": session})` — wire stages
3. `state = TrackerState.empty()` — start with blank smoothing state
4. Per-frame: `observation, state = tracker.process_image(image, frame_number, state)`
5. `session.close()` — release GPU resources when done

### Multi-camera

1. `session = OnnxSession.create(OnnxSessionConfig(batch_size=N, ...))` — allocate once for all N cameras
2. `tracker = Tracker.create(tracker_config, sessions={"onnx": session})`
3. `states = {cam_id: TrackerState.empty() for cam_id in camera_ids}`
4. Per-frame: `observations, states = tracker.process_batch(images, frame_number, states)`
5. `session.close()`

The session is created with `batch_size=N` so models are loaded to accept `(N, 3, H, W)` tensors. The tracker and stage tree are identical in both cases — only the call site differs.
