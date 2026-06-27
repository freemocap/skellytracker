# skellytracker — New Architecture

This folder documents the redesigned skellytracker architecture. The core shift is making the **Top-Down pose estimation paradigm** a first-class compositional primitive: an image flows through a `Tracker`, which runs one or more `DetectionStage`s (each an optional `ObjectDetector` crop followed by one or more `KeypointDetector`s), and produces a structured `Observation`. Stages can be nested hierarchically (body → face, body → hands), and the `Tracker` carries explicit per-frame `TrackerState` for temporal smoothing.

## Component Map

| Component | Role | Doc |
|-----------|------|-----|
| **Tracker** | Pipeline orchestrator; owns stages and state | [01-tracker.md](./01-tracker.md) |
| **ObjectDetector / KeypointDetector** | Primitive detection units | [02-detectors.md](./02-detectors.md) |
| **DetectionStage** | Composes detectors; supports hierarchical nesting | [03-detection-stage.md](./03-detection-stage.md) |
| **Observation** | Per-frame structured output | [04-observation.md](./04-observation.md) |
| **Session** | GPU/CPU resource manager | [05-session.md](./05-session.md) |
| **TrackerState** | Temporal smoothing state (passed in/out) | [06-tracker-state.md](./06-tracker-state.md) |
| **Annotator / DataStore / DemoManager** | Supporting objects | [07-supporting-objects.md](./07-supporting-objects.md) |

## Data Flow

```
                        ┌─────────────────────────────────────────┐
                        │                 Tracker                  │
                        │                                          │
  image ──────────────► │  ┌──────────────────────────────────┐   │
  state ──────────────► │  │        DetectionStage (body)      │   │
                        │  │                                   │   │
                        │  │  ObjectDetector ──► bbox crop     │   │
                        │  │       ↓                           │   │
                        │  │  KeypointDetector(s) ──► keypts   │   │
                        │  │       ↓                           │   │
                        │  │  ┌─────────────────────────────┐  │   │
                        │  │  │  DetectionStage (face)      │  │   │
                        │  │  │  (child; receives body crop) │  │   │
                        │  │  │  ObjectDetector ──► bbox     │  │   │
                        │  │  │  KeypointDetector ──► keypts │  │   │
                        │  │  └─────────────────────────────┘  │   │
                        │  └──────────────────────────────────┘   │
                        │                                          │
                        │  (repeat for additional top-level stages) │
                        │                                          │
  observation ◄──────── │  merge all stage outputs → Observation   │
  updated state ◄─────  │  update TrackerState                     │
                        └─────────────────────────────────────────┘
```

## Key Design Decisions

**Top-Down is the default paradigm.** Every `DetectionStage` starts with an optional `ObjectDetector` that crops the region of interest. If no `ObjectDetector` is provided, the full image is used (equivalent to a "global" stage, still composable with the rest of the pipeline).

**Hierarchical stages by nesting, not flat lists.** Child stages receive the parent's cropped image and keypoint context. This makes face/hand specialization a structural property of the stage tree, not ad-hoc logic inside a detector.

**State is explicit and external.** `TrackerState` is passed into `process_image()` and returned updated. The Tracker itself is stateless between calls — it never mutates itself. This makes the state inspectable, serializable, and testable independently.

**Session is a top-level concern.** GPU sessions (ONNX Runtime, MediaPipe handles) live in a `Session` object that is created once and passed to detectors. Detectors do not own resource lifecycle.

**Observation is the contract.** Downstream consumers (freemocap triangulation, annotators, data stores) only depend on `Observation`. Tracker internals are opaque.

## Comparison with Current Architecture

| Current | New |
|---------|-----|
| `BaseDetector` — monolithic, tracker-specific | `ObjectDetector` + `KeypointDetector` — separate primitives |
| `BaseTracker` — composes 1 detector + annotator + recorder | `Tracker` — composes N hierarchical `DetectionStage`s |
| Session logic buried inside detector | `Session` is a first-class top-level object |
| ROI smoothing state hidden in session fields | `TrackerState` is explicit, passed in/out |
| `BaseRecorder` | `DataStore` |
| Demo logic inline in `BaseTracker` | `DemoManager` is a separate object |
