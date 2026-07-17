# Temporal Processing

> **This is a pre-implementation design doc.** The bbox-policy section below (`smooth_center`/`smooth_size`, the simpler `BBoxPolicy`) predates the actual implementation and no longer matches the code — see [11-bbox-policy-guide.md](./11-bbox-policy-guide.md) for the current, as-built API (`smooth_bbox`, `keypoint_tracked_bbox`, shrink-rate clamping, etc). The keypoint-smoothing (one-euro/Kalman) sections are still broadly accurate.

Temporal processing is the logic inside `DetectionStage.process()` that uses `StageState` to carry information across frames. It has two sub-concerns: a pre-detection **bbox policy** that decides whether to re-run the object detector, and post-detection **output smoothing** that filters noisy bbox and keypoint measurements. Neither concern belongs to the detectors themselves — detectors are stateless and frame-local. The `TrackerState` exists precisely to give temporal processing the memory it needs.

## Why It Exists

Detection is expensive and noisy. Running the object detector every frame wastes compute when the subject is moving slowly; single-frame detections jitter even when the subject is still. Temporal processing addresses both:

- **Bbox reuse** skips the object detector on frames where the current bbox is still good, trading a small amount of freshness for significant latency reduction.
- **Bbox smoothing** stabilises the crop region across frames, so keypoint detectors see a steady image rather than a jittering window.
- **Keypoint smoothing** reduces per-frame measurement noise while adapting to fast motion, so the pose output is stable without introducing perceptible lag.

## Execution Order

The full execution order inside `DetectionStage.process()`, including temporal steps:

```
DetectionStage.process(image, frame_number, stage_state):

  1. BBoxPolicy.should_redetect(frame_number, stage_state)
       → True:  run ObjectDetector on image → raw_bbox
       → False: reuse smoothed bbox from stage_state.bbox_state

  2. Apply bbox smoothing (EMA) to raw_bbox (or reused bbox)
       → smoothed_bbox, updated bbox_state

  3. Crop image to smoothed_bbox

  4. Run each KeypointDetector on crop → raw_keypoints

  5. Translate raw_keypoints back to full-frame coordinates

  6. Apply keypoint filter (one-euro) to raw_keypoints
       → smoothed_keypoints, updated keypoint_states

  7. Run child DetectionStages (passing crop + parent keypoints as context)

  8. Return (StageObservation with smoothed outputs, updated StageState)
```

Steps 1–2 are pre-detection temporal logic. Steps 6–7 are post-detection temporal logic. Steps 3–5 are unchanged detection steps.

## BBoxPolicy — Conditional Object Redetection

`BBoxPolicy` is the named object that governs when the `ObjectDetector` runs. The default is to redetect every frame, which is identical to having no policy.

```python
@dataclass
class BBoxPolicy:
    redetect_interval: int = 1  # rerun every N frames; 1 = every frame
    fitness_checks: list[BBoxFitnessCheck] = field(default_factory=list)

    def should_redetect(self, frame_number: int, stage_state: StageState) -> bool:
        if stage_state.bbox_state.last_detection_frame is None:
            return True  # no previous detection; must run
        frames_since = frame_number - stage_state.bbox_state.last_detection_frame
        if frames_since >= self.redetect_interval:
            return True
        return any(check.fails(stage_state) for check in self.fitness_checks)
```

### BBoxFitnessCheck

Fitness checks force redetection when the current bbox is no longer reliable, regardless of the frame interval. They read the previous frame's keypoints and bbox from `StageState`.

| Check | Triggers redetection when |
|---|---|
| `MinKeypointVisibility(threshold)` | mean keypoint visibility falls below `threshold` |
| `KeypointsWithinBBoxRatio(threshold)` | fewer than `threshold`% of keypoints are inside the current bbox |
| `MaxFramesWithoutRedetect(n)` | absolute cap — redetect at least every `n` frames no matter what |

`MinKeypointVisibility` catches occlusion and partial exits from frame. `KeypointsWithinBBoxRatio` catches cases where the person moved faster than the redetect interval anticipated. `MaxFramesWithoutRedetect` is a safety valve when `redetect_interval` is set aggressively.

### Config

```python
class BBoxPolicyConfig(BaseModel):
    redetect_interval: int = 1
    fitness_checks: list[BBoxFitnessCheckConfig] = []
```

Added to `DetectionStageConfig`:

```python
bbox_policy: BBoxPolicyConfig = BBoxPolicyConfig()  # default: redetect every frame
```

## BBox Smoothing (EMA)

Bbox smoothing is applied after the redetection decision, so the crop fed into the keypoint detector is always smoothed regardless of whether the bbox came from fresh detection or reuse. It uses `BBoxSmoothingState` (see `06-tracker-state.md`).

EMA is applied independently to the center coordinates and the diagonal size:

```python
def apply_bbox_ema(
    raw_bbox: BoundingBox,
    state: BBoxSmoothingState,
) -> tuple[BoundingBox, BBoxSmoothingState]:
    if state.smooth_center is None:
        return raw_bbox, replace(state, smooth_center=raw_bbox.center, smooth_size=raw_bbox.diagonal)
    alpha = state.alpha
    cx = alpha * state.smooth_center[0] + (1 - alpha) * raw_bbox.center[0]
    cy = alpha * state.smooth_center[1] + (1 - alpha) * raw_bbox.center[1]
    size = alpha * state.smooth_size + (1 - alpha) * raw_bbox.diagonal
    smoothed = BoundingBox.from_center_size(cx, cy, size)
    return smoothed, replace(state, smooth_center=(cx, cy), smooth_size=size)
```

Config: `alpha` in `BBoxSmoothingState` (0–1; higher = more lag, more jitter reduction).

## Keypoint Smoothing — One-Euro Filter

The one-euro filter adapts its low-pass cutoff frequency based on the speed of motion: during slow movement the cutoff is low (aggressive smoothing, minimal jitter); during fast movement the cutoff rises (less smoothing, minimal lag). This makes it well-suited to pose estimation, where both slow controlled poses and fast dynamic movements occur.

```python
def apply_one_euro(
    raw_keypoints: Keypoints,
    state: KeypointSmoothingState,
    dt: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
) -> tuple[Keypoints, KeypointSmoothingState]:
    # vectorised over all N points simultaneously
    # returns filtered keypoints and updated state
    ...
```

The filter operates on x and y coordinates independently (and optionally z). `KeypointSmoothingState` holds `x_prev`, `dx_prev`, `y_prev`, `dy_prev` — the previous filtered values and derivative estimates — one entry per keypoint.

### Config

```python
class KeypointSmoothingConfig(BaseModel):
    kind: Literal["one_euro"] = "one_euro"
    min_cutoff: float = 1.0   # Hz; controls jitter at low speed
    beta: float = 0.0         # speed coefficient; controls lag during fast motion
    d_cutoff: float = 1.0     # derivative low-pass cutoff
```

Added to `DetectionStageConfig`:

```python
keypoint_smoothing: KeypointSmoothingConfig | None = None  # None = no smoothing
```

### Kalman Filter

A Kalman filter is a natural alternative, particularly when occlusion is common: the prediction step can extrapolate keypoint positions across frames where detection fails, rather than holding the last known value. The `kind` discriminator on `KeypointSmoothingConfig` is intentionally designed to accommodate this — `kind: "kalman"` with Kalman-specific parameters can be added later without changing callers. One-euro is the initial implementation because it requires no motion model and has fewer parameters to tune.

## Interaction with TrackerState

`StageState` carries all temporal processing state for one stage:

- `bbox_state.last_detection_frame` — updated to `frame_number` whenever `BBoxPolicy` runs the detector. Used by `should_redetect()` to compute `frames_since`.
- `bbox_state.smooth_center`, `bbox_state.smooth_size` — updated by EMA after every frame.
- `keypoint_states[i]` — updated by the one-euro filter after each keypoint detector runs.

All state is returned as a new value; nothing is mutated in place. See `06-tracker-state.md` for full structure.

## Defaults and Opt-In Behaviour

| Config | Default | Effect |
|---|---|---|
| `bbox_policy.redetect_interval` | `1` | Redetect every frame — identical to no policy |
| `bbox_policy.fitness_checks` | `[]` | No fitness gating |
| `keypoint_smoothing` | `None` | Raw keypoints passed through unchanged |
| `bbox_state.alpha` | `0.5` | Moderate EMA smoothing; set to `0.0` to disable |

A tracker with all defaults applied behaves identically to a tracker with no temporal processing layer. Features are enabled by setting non-default values in config.
