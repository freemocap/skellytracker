# ObjectDetector and KeypointDetector

These are the two primitive detection units. They are kept separate because they answer different questions: the `ObjectDetector` asks "where is the subject?" and the `KeypointDetector` asks "what is the subject doing?" Splitting them makes it easy to swap detection backends (e.g., replace YOLO with a different person detector) without touching the keypoint estimation logic, and vice versa.

## ObjectDetector

Runs on an image (full frame or a cropped region) and returns one or more bounding boxes. The bounding boxes are what gets passed to the `KeypointDetector` and to any child `DetectionStage`s.

```python
@dataclass
class ObjectDetector(ABC):
    config: ObjectDetectorConfig
    session: Session

    @abstractmethod
    def detect(
        self,
        image: NDArray[np.uint8],
    ) -> list[BoundingBox]:
        ...

    @classmethod
    def create(cls, config: ObjectDetectorConfig, session: Session) -> "ObjectDetector":
        ...
```

`BoundingBox` carries the pixel coordinates of the detected region plus a confidence score. When no `ObjectDetector` is present in a stage, the full image is treated as a single implicit bounding box.

**Example implementations:** YOLO-based person detector, face detector, hardcoded full-frame crop, bounding box derived from a parent stage's keypoints (e.g., compute a tight crop around detected wrist keypoints to pass to a hand detector).

## KeypointDetector

Runs on a cropped image (or full image if no crop was applied) and returns a set of named keypoints with pixel coordinates and visibility scores. A `DetectionStage` can contain multiple `KeypointDetector`s — for example, running separate left-hand and right-hand models on independent crops within the same stage.

```python
@dataclass
class KeypointDetector(ABC):
    config: KeypointDetectorConfig
    tracked_object: TrackedObjectDefinition  # from YAML
    session: Session

    @abstractmethod
    def detect(
        self,
        image: NDArray[np.uint8],
        bbox: BoundingBox | None,
    ) -> Keypoints:
        ...

    @classmethod
    def create(cls, config: KeypointDetectorConfig, session: Session) -> "KeypointDetector":
        ...
```

`Keypoints` wraps a `PointCloud` (the existing canonical data primitive: ordered named points with xyz coordinates and visibility scores). Point names and ordering are defined by the YAML `TrackedObjectDefinition` associated with the detector.

**Example implementations:** RTMPose whole-body, MediaPipe body/hand/face, VitPose, brightest-point.

## Shared Conventions

- Both types receive a `Session` at construction time — they do not own GPU resources.
- Both are stateless: no per-frame mutable fields. All temporal state lives in `TrackerState`.
- Both are constructed via `create(config, session)` classmethods.
- Detector configs are Pydantic models; the config fully describes what model to load and how to run it.
