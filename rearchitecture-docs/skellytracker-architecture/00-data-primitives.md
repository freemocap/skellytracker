# Data Primitives

`Keypoints` and `BoundingBox` are the two low-level data primitives everything else is built on. They are internal building blocks — callers work with `Observation`, and these primitives are what the internals use to hold and pass around detection results.

---

## Keypoints

A named array of 3D points with visibility scores. This is what a `KeypointDetector` produces and what `StageObservation` stores. The core invariant is structural coupling: the names tuple and the coordinate array are always the same length and cannot desync.

```python
@dataclass(slots=True)
class Keypoints:
    names: tuple[str, ...]
    xyz: NDArray[np.float64]         # (N, 3) — x, y in pixels; z filled by triangulation
    visibility: NDArray[np.float64]  # (N,)   — confidence scores, 0.0–1.0
```

Undetected points have `nan` coordinates and `0.0` visibility. Downstream code checks `visibility` (or uses `filtered_by_confidence()`) rather than checking for `None`.

Key methods:

- **`xyz_by_name(name)`** / **`xy_by_name(name)`** — name-based coordinate lookup, O(1) via a cached index dict. Used by child stages to compute crops from parent keypoints (e.g., "where is `body.left_wrist`?").
- **`filtered_by_confidence(threshold)`** — returns a same-sized `Keypoints` with low-confidence points set to `nan`, or a smaller `Keypoints` with only high-confidence points.
- **`concatenate(clouds)`** — stacks multiple `Keypoints` into one flat named array. This is how `observation.to_keypoints()` works: it merges all stage keypoints into a single `Keypoints` for freemocap triangulation.
- **`empty(names)`** — factory for a blank `Keypoints` (all `nan`, all zero visibility) sized to a given name list.
- **`slice(start, stop)`** / **`slice_by_names(names)`** — range or name-based subsets.

The `z` column is `0.0` after 2D detection and is filled in by freemocap's multi-camera triangulation step.

---

## BoundingBox

A rectangular region in pixel space, produced by an `ObjectDetector`. It is what gets passed to `KeypointDetector`s and child `DetectionStage`s to define their input crop.

```python
@dataclass
class BoundingBox:
    x1: float          # left edge, pixels
    y1: float          # top edge, pixels
    x2: float          # right edge, pixels
    y2: float          # bottom edge, pixels
    confidence: float  # detection confidence, 0.0–1.0
```

Key properties:

- **`center`** — `(x, y)` midpoint; used by `TrackerState` for bounding box smoothing.
- **`size`** — `(width, height)`; used for adaptive crop scaling.
- **`area`** — scalar; used for sanity checks (e.g., reject implausibly small detections).
- **`to_crop(image)`** — returns the image region bounded by this box, ready to pass to a `KeypointDetector`.
- **`scaled(factor)`** / **`padded(px)`** — expand the box by a scale factor or fixed padding before cropping.

When no `ObjectDetector` is present in a stage, a `BoundingBox` spanning the full image is used implicitly. When `TrackerState` smooths bounding boxes across frames, it operates on the `center` and diagonal `size` of a `BoundingBox`.
