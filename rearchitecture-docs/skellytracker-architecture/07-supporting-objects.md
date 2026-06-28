# Supporting Objects

These three objects sit outside the core detection pipeline and handle the surrounding concerns: visualization, data persistence, and live demo management.

---

## Annotator

The `Annotator` takes an image and an `Observation` and returns the image with bounding boxes, keypoints, and skeleton connections drawn on it. It is mostly detector-agnostic: it reads point names and skeleton connections from YAML-defined schemas, so it can annotate any `Observation` without knowing which specific detectors produced it.

```python
@dataclass
class Annotator:
    config: AnnotatorConfig
    stage_schemas: dict[str, SkeletonSchema]  # YAML-defined schemas keyed by stage name

    def annotate(
        self,
        image: NDArray[np.uint8],
        observation: Observation,
    ) -> NDArray[np.uint8]:
        ...

    @classmethod
    def create(cls, config: AnnotatorConfig) -> "Annotator":
        ...
```

The `AnnotatorConfig` controls visual style: keypoint radius, line thickness, color scheme per stage (e.g., body=green, hands=red/blue, face=yellow), whether to draw bounding boxes, and confidence threshold below which points are not drawn.

Annotation is done per stage: for each `StageObservation`, the annotator looks up the YAML-defined schema for that stage, resolves connection indices from the point names, and draws skeleton edges and keypoints. Bounding boxes from `ObjectDetector`s are drawn separately.

---

## DataStore

The `DataStore` accumulates `Observation`s across frames and serializes them. It replaces the current `BaseRecorder`. The new name reflects that it holds structured data rather than implying a passive recording metaphor.

```python
@dataclass
class DataStore:
    observations: list[Observation] = field(default_factory=list)

    def add(self, observation: Observation) -> None:
        self.observations.append(observation)

    def to_array(self) -> NDArray[np.float64]:
        # shape: (num_frames, num_points, 3)
        ...

    def to_json(self) -> str:
        ...

    def save(self, path: Path, format: Literal["npy", "json"] = "npy") -> None:
        ...
```

The primary output format is a `(frames, points, 3)` numpy array (`.npy`), matching what freemocap expects for triangulation. The JSON format provides a human-readable alternative with point names included.

A `DataStore` is optional — callers that only need live visualization and not persistence can skip it.

---

## DemoManager

The `DemoManager` runs the tracker in a live loop against a webcam or video file. It handles the frame-source setup, the per-frame call to `Tracker.process_image()`, passing results to an `Annotator` for display, and window/playback management.

```python
@dataclass
class DemoManager:
    tracker: Tracker
    annotator: Annotator
    data_store: DataStore | None  # optional; pass to also accumulate frames

    def run_webcam(self, camera_index: int = 0) -> None:
        ...

    def run_video(self, video_path: Path) -> None:
        ...
```

The `DemoManager` owns the frame loop and the OpenCV window. It maintains the `TrackerState` across frames, calls `tracker.process_image()`, calls `annotator.annotate()`, and optionally calls `data_store.add()`. On `q` or window close, it calls `session.close()` and exits cleanly.

The `DemoManager` is intentionally thin — it wires together existing objects rather than containing logic of its own.
