# Tracker

The `Tracker` is the top-level pipeline object. It is the only public entry point for running pose estimation: callers pass in a frame and the current `TrackerState`, and receive an `Observation` and an updated state. Everything else — detection stages, sessions, smoothing — is an implementation detail behind this interface.

A `Tracker` owns a list of `DetectionStage`s. Stages are ordered; each runs in sequence on the input image (or a crop derived from a parent stage). Stages can be nested, so the full pipeline is a tree rather than a flat list.

The `Tracker` does not own mutable state itself. It reads `TrackerState` on each call and returns an updated copy.

## Interface

```python
@dataclass
class Tracker:
    stages: list[DetectionStage]
    session: Session

    def process_image(
        self,
        image: NDArray[np.uint8],
        frame_number: int,
        state: TrackerState,
    ) -> tuple[Observation, TrackerState]:
        ...

    @classmethod
    def create(cls, config: TrackerConfig, session: Session) -> "Tracker":
        ...
```

`process_image` is the single hot-path method. It runs all stages, merges their outputs into an `Observation`, and returns the updated `TrackerState` alongside it.

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

1. `Session.create(config)` — allocate GPU/CPU resources
2. `Tracker.create(tracker_config, session)` — wire stages, detectors reference the session
3. `state = TrackerState.empty()` — start with blank smoothing state
4. Per-frame: `observation, state = tracker.process_image(image, frame_number, state)`
5. `session.close()` — release GPU resources when done
