# DetectionStage

A `DetectionStage` is the compositional unit of the pipeline. It binds together one optional `ObjectDetector` and one or more `KeypointDetector`s into a single step, and can carry child stages that operate on its output. The stage tree is what gives the `Tracker` its hierarchical structure.

## Structure

```python
@dataclass
class DetectionStage:
    object_detector: ObjectDetector | None
    keypoint_detectors: list[KeypointDetector]
    children: list[DetectionStage]
    name: str
```

## Execution

### Single-camera: `run(image, state, context)`

1. **Object detection** (optional): `BBoxPolicy` decides whether to re-run the `ObjectDetector` or reuse the smoothed bbox from the previous frame. The resulting bbox is smoothed via EMA before use. If no detector is present, the full image is used as a single bounding box.
2. **Keypoint detection**: each `KeypointDetector` runs on the image cropped to the smoothed bounding box. Keypoints are translated back to full-frame coordinates, then passed through the keypoint filter (one-euro or none).
3. **Child stages**: each child `DetectionStage` receives the cropped image (and the parent's keypoints as context, e.g., to derive its own crop region) and runs its own detection subtree.
4. **Output**: the stage returns a `StageObservation` containing its bounding boxes, keypoints, and the observations from all child stages.

The pre- and post-detection manipulation steps are described in full in `09-temporal-processing.md`.

### Multi-camera: `run_batch(images, states, context)`

`run_batch` is the batched equivalent that processes N cameras in two GPU calls instead of N. It is the orchestrator for multi-camera inference — the `Session` provides the batch infrastructure, but `DetectionStage` decides what gets batched and when.

```
images: dict[cam_id, NDArray]
states: dict[cam_id, StageState]
```

Execution order within `run_batch`:

1. **Preprocess all cameras** — call `object_detector.preprocess(image)` for each camera. If cameras share a resolution, these can be vectorized across the batch dimension. Returns per-camera `(tensor, metadata)` pairs.
2. **Batch object detection** — stack all N preprocessed tensors into `(N, 3, H, W)`, call `session.run_batched(model_name, stacked)` once, split results back by camera key.
3. **Postprocess + bbox smoothing per camera** — call `object_detector.postprocess(raw, meta)` and apply bbox EMA for each camera independently using its own `StageState`.
4. **Compute crops per camera** — each camera's smoothed bbox determines its own crop region.
5. **Preprocess all crops** — call `keypoint_detector.preprocess(crop)` for each camera.
6. **Batch keypoint detection** — stack all N crop tensors, call `session.run_batched()` once, split results.
7. **Postprocess + keypoint filtering per camera** — decode keypoints, apply one-euro filter using per-camera `StageState`.
8. **Child stages** — children run per-camera (their inputs differ across cameras), calling `run_batch` recursively on their own sub-images.

Top-down pipelines (object detect → crop → keypoint detect) therefore use exactly **two batched GPU calls** per stage, regardless of camera count. Stages with no `ObjectDetector` use one call.

See [10-multi-camera-batching.md](./10-multi-camera-batching.md) for the full design, including parallelism strategies for pre/postprocessing and the MediaPipe fallback path.

## Hierarchical Example: Body → Face

```
DetectionStage("body")
  ObjectDetector: PersonDetector       # finds person in full frame
  KeypointDetector: BodyPoseDetector   # estimates 17 body keypoints
  children:
    DetectionStage("face")
      ObjectDetector: FaceRegionFromBody  # derives face crop from head keypoints
      KeypointDetector: FaceKeypointDetector  # estimates 106 face keypoints
    DetectionStage("hands")
      ObjectDetector: HandRegionFromBody  # derives hand crops from wrist keypoints
      KeypointDetector: HandKeypointDetector (×2, right + left)
```

The parent stage's bounding boxes and keypoints are available to child stages when computing their crops. This is how the existing "crop hands/face from body pose" logic is represented structurally rather than as ad-hoc code inside a detector.

## Multiple Top-Level Stages

A `Tracker` can have multiple top-level stages, each running independently on the full image. This supports cases like running a separate body detector and a separate instrument detector in the same frame, without either being a child of the other.

## Stage Outputs and Merging

Each stage produces a `StageObservation` identified by its `name`. The `Tracker` collects all stage observations (including nested children) and merges them into a single `Observation`. The stage name is the key that downstream consumers use to pull out specific keypoints or bounding boxes.

Point names within a stage come from the YAML-defined schema for each `KeypointDetector`. The stage may apply a name prefix (e.g., `"face."`) to avoid collisions when merging into the top-level `Observation`.
