# ObjectDetector and KeypointDetector

These are the two primitive detection units. They are kept separate because they answer different questions: the `ObjectDetector` asks "where is the subject?" and the `KeypointDetector` asks "what is the subject doing?" Splitting them makes it easy to swap detection backends (e.g., replace YOLO with a different person detector) without touching the keypoint estimation logic, and vice versa.

## The Three-Step Contract

Every detector — object or keypoint, ONNX or MediaPipe or OpenCV — exposes three methods that together replace `detect`:

```python
def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray, Metadata]:
    """Prepare the image for inference. Returns a float32 tensor and
    any metadata needed by postprocess (e.g. letterbox ratio, crop center)."""
    ...

def infer(self, tensor: NDArray, session: Session) -> Any:
    """Run the model. For ONNX detectors this calls session.run_batched();
    for MediaPipe it calls mp_solution.process(); for OpenCV it calls
    cv2.aruco.detectMarkers() etc. Returns raw model output."""
    ...

def postprocess(self, raw: Any, metadata: Metadata) -> list[BoundingBox] | Keypoints:
    """Decode raw model output into the detector's result type using the
    metadata from preprocess (e.g. untransform coordinates, run NMS)."""
    ...
```

`detect` remains as a convenience wrapper for single-image single-call use:

```python
def detect(self, image: NDArray[np.uint8], session: Session) -> list[BoundingBox] | Keypoints:
    tensor, meta = self.preprocess(image)
    raw = self.infer(tensor, session)
    return self.postprocess(raw, meta)
```

`DetectionStage.run_batch()` calls these steps separately across all cameras: preprocess all N images, call `session.run_batched()` once with the stacked batch, then postprocess each camera's output independently. See [10-multi-camera-batching.md](./10-multi-camera-batching.md).

### Metadata

`Metadata` is a small typed dataclass defined per detector — not `Any` at runtime. It carries only what `postprocess` needs that can't be reconstructed from the raw output alone:

| Detector | Metadata fields |
|----------|----------------|
| YOLOX | `ratio: float`, `original_size: tuple[int, int]` |
| RTMPose | `center: NDArray`, `scale: NDArray` |
| MediaPipe | *(empty — no coordinate untransform needed)* |
| Charuco / ArUco | *(empty — OpenCV returns image-space coords directly)* |
| BrightestPoint | *(empty)* |

Steps may be empty (identity) when the backend handles everything internally. For MediaPipe, `preprocess` is a BGR→RGB conversion, `infer` calls `mp_solution.process()`, and `postprocess` extracts landmark coordinates from the result object.

---

## ObjectDetector

Runs on an image (full frame or a cropped region) and returns one or more bounding boxes. The bounding boxes are what gets passed to the `KeypointDetector` and to any child `DetectionStage`s.

```python
@dataclass
class ObjectDetector(ABC):
    config: ObjectDetectorConfig
    session: Session

    @abstractmethod
    def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray, Metadata]: ...

    @abstractmethod
    def infer(self, tensor: NDArray, session: Session) -> Any: ...

    @abstractmethod
    def postprocess(self, raw: Any, metadata: Metadata) -> list[BoundingBox]: ...

    def detect(self, image: NDArray[np.uint8]) -> list[BoundingBox]:
        tensor, meta = self.preprocess(image)
        raw = self.infer(tensor, self.session)
        return self.postprocess(raw, meta)

    @classmethod
    def create(cls, config: ObjectDetectorConfig, session: Session) -> "ObjectDetector": ...
```

`BoundingBox` carries the pixel coordinates of the detected region plus a confidence score (see [data primitives](./00-data-primitives.md)). When no `ObjectDetector` is present in a stage, the full image is treated as a single implicit bounding box.

**Example implementations:** YOLOX person detector, face detector, hardcoded full-frame crop, bounding box derived from a parent stage's keypoints.

---

## KeypointDetector

Runs on a cropped image (or full image if no crop was applied) and returns a set of named keypoints with pixel coordinates and visibility scores. A `DetectionStage` can contain multiple `KeypointDetector`s.

```python
@dataclass
class KeypointDetector(ABC):
    config: KeypointDetectorConfig
    session: Session

    @abstractmethod
    def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray, Metadata]: ...

    @abstractmethod
    def infer(self, tensor: NDArray, session: Session) -> Any: ...

    @abstractmethod
    def postprocess(self, raw: Any, metadata: Metadata) -> Keypoints: ...

    def detect(self, image: NDArray[np.uint8]) -> Keypoints:
        tensor, meta = self.preprocess(image)
        raw = self.infer(tensor, self.session)
        return self.postprocess(raw, meta)

    @classmethod
    def create(cls, config: KeypointDetectorConfig, session: Session) -> "KeypointDetector": ...
```

`Keypoints` is the low-level data primitive for named points (see [data primitives](./00-data-primitives.md)). Point names and ordering are defined by the YAML schema for each detector.

**Example implementations:** RTMPose whole-body, MediaPipe body/hand/face, brightest-point, Charuco.

---

## Shared Conventions

- Both types receive a `Session` at construction time — they do not own GPU resources.
- Both are stateless: no per-frame mutable fields. All temporal state lives in `TrackerState`.
- Both are constructed via `create(config, session)` classmethods.
- Detector configs are Pydantic models.
- `preprocess` and `postprocess` may be empty (identity/passthrough) for backends that handle everything inside `infer`.
