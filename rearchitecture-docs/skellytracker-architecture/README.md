# skellytracker — New Architecture

This folder documents the redesigned skellytracker architecture. The core shift is making the **Top-Down pose estimation paradigm** a first-class compositional primitive: an image flows through a `Tracker`, which runs one or more `DetectionStage`s (each an optional `ObjectDetector` crop followed by one or more `KeypointDetector`s), and produces a structured `Observation`. Stages can be nested hierarchically (body → face, body → hands), and the `Tracker` carries explicit per-frame `TrackerState` for temporal smoothing.

## Component Map

| Component | Role | Doc |
|-----------|------|-----|
| **Keypoints / BoundingBox** | Low-level data primitives | [00-data-primitives.md](./00-data-primitives.md) |
| **Tracker** | Pipeline orchestrator; owns stages and state | [01-tracker.md](./01-tracker.md) |
| **ObjectDetector / KeypointDetector** | Primitive detection units | [02-detectors.md](./02-detectors.md) |
| **DetectionStage** | Composes detectors; supports hierarchical nesting | [03-detection-stage.md](./03-detection-stage.md) |
| **Observation** | Per-frame structured output | [04-observation.md](./04-observation.md) |
| **Session** | GPU/CPU resource manager and batch coordinator | [05-session.md](./05-session.md) |
| **TrackerState** | Temporal smoothing state (passed in/out) | [06-tracker-state.md](./06-tracker-state.md) |
| **Annotator / DataStore / DemoManager / process_video** | Supporting objects | [07-supporting-objects.md](./07-supporting-objects.md) |
| **ONNX Batching and CoreML** | Dynamic batch surgery and CoreML compatibility | [08-onnx-batching-and-coreml.md](./08-onnx-batching-and-coreml.md) |
| **Temporal Processing** | BBox policy, EMA smoothing, keypoint filtering | [09-temporal-processing.md](./09-temporal-processing.md) |
| **Multi-Camera Batching** | Batched inference across cameras; pre/infer/post split | [10-multi-camera-batching.md](./10-multi-camera-batching.md) |
| **BBox Policy (current API)** | As-built bbox policy/smoothing API and freemocap RTMPose integration guide | [11-bbox-policy-guide.md](./11-bbox-policy-guide.md) |
| **Multi-Person Tracking** | Cross-frame identity tracking for N people, single camera (IoU + keypoint association) | [12-multi-person-tracking.md](./12-multi-person-tracking.md) |

## Data Flow

### Single-camera

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

### Multi-camera (batched)

```
  {"cam0": image, "cam1": image, ...} ──────────────────────────────► Tracker.process_batch()
  {"cam0": state, "cam1": state, ...} ──────────────────────────────►
                                                                        │
                                                              DetectionStage.run_batch()
                                                                        │
                                              ┌─────────────────────────────────────────┐
                                              │  preprocess  (per-camera, vectorized)   │
                                              │      ↓                                  │
                                              │  Session.run_batched()  [one GPU call]  │
                                              │      ↓                                  │
                                              │  postprocess (per-camera, vectorized)   │
                                              │      ↓                                  │
                                              │  temporal processing (per-camera)       │
                                              └─────────────────────────────────────────┘
                                                                        │
  {"cam0": obs, "cam1": obs, ...}   ◄─────────────────────────────────┘
  {"cam0": state, "cam1": state, ...} ◄───────────────────────────────┘
```

See [10-multi-camera-batching.md](./10-multi-camera-batching.md) for full detail.

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
