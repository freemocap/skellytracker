# Observation

An `Observation` is the per-frame output of the `Tracker`. It is the contract between the tracking pipeline and everything downstream: annotators, data stores, freemocap triangulation, visualization tools. Nothing downstream should need to know which detectors produced the data.

## Structure

```python
@dataclass
class Observation:
    frame_number: int
    image_size: tuple[int, int]         # (height, width) in pixels
    timestamp: float | None             # wall-clock time, if available
    stages: dict[str, StageObservation] # keyed by stage name
```

```python
@dataclass
class StageObservation:
    name: str
    bounding_boxes: list[BoundingBox]   # from ObjectDetector (may be empty)
    keypoints: PointCloud               # merged from all KeypointDetectors in the stage
    children: dict[str, StageObservation]
```

The `PointCloud` is the existing canonical data primitive (named points + xyz coordinates + visibility scores). Point names within a `StageObservation` follow the YAML `TrackedObjectDefinition` for that stage's `KeypointDetector`s, with prefixes applied to avoid collisions.

## Canonical Access

Downstream code that wants all keypoints as a flat array calls `observation.to_point_cloud()`, which concatenates all stage `PointCloud`s (in a stable, config-determined order) into a single named `PointCloud`. This is the form passed to freemocap for triangulation.

```python
cloud: PointCloud = observation.to_point_cloud()
# cloud.names: ("body.nose", "body.left_shoulder", ..., "face.left_eye", ...)
# cloud.xyz: (N, 3) array
# cloud.visibility: (N,) array
```

## What Observations Carry

- **Bounding boxes**: pixel coordinates + confidence, per stage. Used by annotators and for debugging.
- **Keypoints**: named 2D coordinates (x, y, 0.0 for z until triangulated) + visibility scores. The z column is filled in later by freemocap's triangulation step.
- **Frame metadata**: frame number, image size, timestamp. Needed for temporal alignment across cameras.
- **Stage hierarchy**: the tree structure is preserved so annotators can draw stage-specific colors/styles and downstream code can pull out specific sub-observations by name.

## Absence of Detection

If no object was detected in a stage, the stage's `bounding_boxes` list is empty and the `keypoints` `PointCloud` has `nan` for all coordinates and `0.0` for all visibility scores. Downstream code checks `PointCloud.visibility` (or `filtered_by_confidence()`) rather than checking for `None`.

## Naming Convention

Point names in the merged `PointCloud` follow dot-notation: `<stage_name>.<point_name>`. This matches the existing YAML-driven `TrackedObjectDefinition` composition pattern (where composite definitions prefix sub-definitions). Stage names are set in the `TrackerConfig` and must be unique within a tracker.
