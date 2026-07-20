# TrackerState

`TrackerState` holds all the temporal data that must persist across frames for smoothing to work: bounding box positions, filter coefficients, and any per-point state needed by keypoint filters. It is passed into `Tracker.process_image()` and returned updated — the `Tracker` never mutates it in place.

## Why Explicit State

In the current architecture, smoothing state (EMA buffers for ROI centers, adaptive hand-size memory) lives as mutable fields on the ONNX `Session` object. This works but has two costs:

1. **Not inspectable**: the state is buried inside the session, making it hard to log, debug, or snapshot.
2. **Not resumable**: to resume tracking after a pause (e.g., camera disconnection), you'd need to reconstruct the session state from scratch.

Making state explicit lets you serialize it, resume from it, inspect it between frames, and test the smoothing logic independently of the detectors.

## Structure

```python
@dataclass
class TrackerState:
    stage_states: dict[str, StageState]  # keyed by stage name

    @classmethod
    def empty(cls, tracker: Tracker) -> "TrackerState":
        # constructs blank state matching the tracker's stage tree
        ...
```

```python
@dataclass
class StageState:
    bbox_state: BBoxSmoothingState | None   # if the stage has an ObjectDetector
    keypoint_states: list[KeypointSmoothingState]  # one per KeypointDetector
    child_states: dict[str, StageState]     # mirrors the stage tree
```

## BBoxSmoothingState

Tracks the smoothed bounding box across frames. Typically an exponential moving average (EMA) on the center coordinates and diagonal size, preventing jitter from single-frame detection noise. Also records when the object detector last ran so that `BBoxPolicy` can decide whether to redetect on the current frame (see `09-temporal-processing.md`).

```python
@dataclass
class BBoxSmoothingState:
    smooth_center: tuple[float, float] | None
    smooth_size: float | None
    alpha: float  # EMA decay factor
    last_detection_frame: int | None  # frame number of most recent ObjectDetector run
```

## KeypointSmoothingState

Per-point state for a one euro filter (or similar adaptive filter) on keypoint coordinates. One euro filters adapt their cutoff frequency based on the speed of motion, reducing lag during fast motion and jitter during slow motion.

```python
@dataclass
class KeypointSmoothingState:
    # one euro filter coefficients, one per point
    x_prev: NDArray[np.float64]   # (N,) last filtered x values
    dx_prev: NDArray[np.float64]  # (N,) last derivative estimates
    # ... additional filter state
```

## Serialization

`TrackerState` is serializable to/from a dict (JSON-compatible). This supports:

- **Resumption**: save state at frame N, resume at frame N+1 without smoothing artifacts.
- **Debugging**: log state at any frame to understand why the filter is producing a particular output.
- **Testing**: inject a known state and verify the filter output deterministically.

## Interaction with Tracker

```python
state = TrackerState.empty(tracker)

for frame_number, image in frame_source:
    observation, state = tracker.process_image(image, frame_number, state)
    # state is now updated with smoothed bboxes and filter coefficients for this frame
```

The `Tracker` passes each stage's state slice down to the relevant `DetectionStage`, which applies smoothing and returns the updated slice. The updated slices are assembled back into a new `TrackerState` and returned.
