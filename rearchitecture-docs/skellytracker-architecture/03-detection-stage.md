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

When a `DetectionStage` runs:

1. **Object detection** (optional): the `ObjectDetector` runs on the incoming image and returns bounding boxes. If no detector is present, the full image is used as a single bounding box.
2. **Keypoint detection**: each `KeypointDetector` runs on the image cropped to the bounding box.
3. **Child stages**: each child `DetectionStage` receives the cropped image (and the parent's keypoints as context, e.g., to derive its own crop region) and runs its own detection subtree.
4. **Output**: the stage returns a `StageObservation` containing its bounding boxes, keypoints, and the observations from all child stages.

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
