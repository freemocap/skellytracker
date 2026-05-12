# SkellyTracker Rust Port — Charuco Milestone Design

## Summary

Port the core SkellyTracker pose-estimation architecture to Rust, starting with the Charuco tracker as the first milestone. The Rust crate lives alongside the Python package at `skellytracker-rust/`. This initial milestone proves the full pipeline: webcam capture → Charuco detection → annotation → display.

## Goal

A standalone Rust binary (`demo.exe` / `demo`) that opens a webcam, runs Charuco board detection on each frame using `opencv::aruco`, draws annotated results with fading corner trails and Aruco marker boxes, and displays in an OpenCV window. Uses the same YAML point-definition format as the Python tracker so point names and connections stay in sync.

## Non-goals

- No MediaPipe, RTMPose, or CompositeGPU trackers (future milestones)
- No video file processing (future milestone)
- No 3D triangulation or calibration (lives in FreeMoCap ecosystem, not skellytracker)
- No Python bindings / PyO3 (pure Rust)

---

## Architecture

### Crate structure

```
skellytracker-rust/
├── Cargo.toml
├── src/
│   ├── lib.rs                          # crate root, re-exports public API
│   ├── core/
│   │   ├── mod.rs
│   │   ├── point_cloud.rs              # PointCloud struct
│   │   ├── observation.rs             # Observation struct + TrackerKind + ObservationPayload enum
│   │   ├── tracked_object_definition.rs  # TrackedObjectDefinition (serde from YAML)
│   │   ├── traits.rs                    # Detect, Annotate, Record traits
│   │   └── tracker.rs                   # Tracker struct (composes Box<dyn Detect> + Box<dyn Annotate> + Recorder)
│   ├── trackers/
│   │   ├── mod.rs
│   │   └── charuco/
│   │       ├── mod.rs
│   │       ├── detector.rs              # CharucoDetector impl Detect
│   │       ├── annotator.rs             # CharucoAnnotator impl Annotate
│   │       ├── config.rs                # CharucoDetectorConfig, CharucoAnnotatorConfig (serde)
│   │       ├── board.rs                 # CharucoBoardDefinition
│   │       ├── observation.rs           # Charuco-specific observation construction
│   │       └── charuco_tracked_object.yaml  # point names + connections
│   ├── io/
│   │   ├── mod.rs
│   │   ├── webcam.rs                    # WebcamCapture (opencv::videoio)
│   │   ├── video.rs                     # Video file I/O (stub for future)
│   │   └── recorder.rs                  # Recorder: collects + serializes to JSON / .npy
│   └── bin/
│       └── demo.rs                      # Webcam demo binary
└── tests/
    ├── charuco_detection_test.rs
    └── fixtures/
        └── charuco_test_image.jpg
```

### Design principles

1. **Many small files.** Each struct gets its own file. No mega-modules.
2. **No abbreviations.** Full words everywhere — `observation`, not `obs`; `history`, not `hist`; `configuration`, not `config` in variable names (but `Config` as a type suffix is acceptable).
3. **Traits for polymorphism, not inheritance.** Equivalent of Python ABCs. `Box<dyn Detect>` enables runtime tracker switching in the demo without touching dispatch code when adding new trackers.
4. **YAML files co-located with tracker code.** Each tracker's `names_and_connections/` YAML lives inside that tracker's directory, not in a separate top-level folder.
5. **`enum` for tracker-specific observation payloads.** The common `Observation` struct holds shared fields; tracker-specific data goes in `ObservationPayload` variants. Exhaustiveness checking guards against unhandled tracker types.

---

## Core data types

### PointCloud (`core/point_cloud.rs`)

The canonical data primitive for tracked landmarks. Mirrors the Python `PointCloud` dataclass.

```rust
pub struct PointCloud {
    names: Vec<String>,                          // ordered, i-th name ↔ i-th row
    xyz: ndarray::Array2<f64>,                   // (N, 3), z=0 for 2D-only
    visibility: ndarray::Array1<f64>,             // (N,), 0.0 = not detected
    name_to_index: HashMap<String, usize>,        // built in constructor
}
```

Key methods:
- `new(names: Vec<String>) -> Self` — all-NaN, zero visibility
- `index_of(name: &str) -> Option<usize>`
- `coordinates_by_name(name: &str) -> Option<[f64; 3]>`
- `slice_by_names(names: &[&str]) -> PointCloud` — data copy
- `filtered_by_confidence(threshold: f64, fill_with_nans: bool) -> PointCloud`
- `xy_view() -> ndarray::ArrayView2<f64>` — zero-copy (N, 2) view

### Observation (`core/observation.rs`)

```rust
pub struct Observation {
    pub frame_number: u64,
    pub tracker_kind: TrackerKind,
    pub points: PointCloud,
    pub payload: ObservationPayload,
}

pub enum TrackerKind {
    Charuco,
    // Future: MediaPipe, RtmPose, CompositeGpu,
}

pub enum ObservationPayload {
    Charuco {
        all_charuco_ids: Vec<i32>,
        all_aruco_ids: Vec<i32>,
        detected_charuco_corner_ids: Option<Vec<i32>>,
        detected_charuco_corners: Option<Vec<[f64; 2]>>,
        detected_aruco_marker_ids: Option<Vec<i32>>,
        detected_aruco_marker_corners: Option<Vec<[[f64; 2]; 4]>>,
        // Board pose (computed post-detection)
        board_rotation_vector: Option<[f64; 3]>,
        board_translation_vector: Option<[f64; 3]>,
        detected_charuco_corners_in_camera_coordinates: Option<Vec<[f64; 3]>>,
    },
}
```

### TrackedObjectDefinition (`core/tracked_object_definition.rs`)

Replaces Python's Pydantic `TrackedObjectDefinition`. Loaded from YAML via `serde_yaml`.

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct TrackedObjectDefinition {
    pub name: String,
    pub tracker_type: String,
    #[serde(default)]
    pub landmark_schema: String,
    pub tracked_points: Vec<String>,
    #[serde(default)]
    pub connections: Vec<(String, String)>,
}
```

Methods:
- `from_yaml(path: &Path) -> Result<Self>` — load, deserialize, validate
- `connection_indices(&self) -> Result<Vec<(usize, usize)>>` — resolve name-pairs to array indices
- `empty_point_cloud(&self) -> PointCloud` — factory for all-NaN cloud

The YAML composition system (`composed_of`) used by MediaPipe Holistic is deferred to the MediaPipe milestone. Charuco YAMLs are flat.

### Traits (`core/traits.rs`)

```rust
pub trait Detect {
    fn detect(&self, frame_number: u64, image: &Mat) -> Result<Observation>;
    fn tracked_object_definition(&self) -> &TrackedObjectDefinition;
}

pub trait Annotate {
    fn annotate(&mut self, image: &Mat, observation: &Observation) -> Result<Mat>;
}

pub trait Record {
    fn add_observation(&mut self, observation: Observation);
    fn clear(&mut self);
    fn observation_count(&self) -> usize;
}
```

### Tracker (`core/tracker.rs`)

```rust
pub struct Tracker {
    pub detector: Box<dyn Detect>,
    pub annotator: Box<dyn Annotate>,
    pub recorder: Box<dyn Record>,
}

impl Tracker {
    /// Run detection and optionally record the observation
    pub fn process_image(
        &mut self,
        frame_number: u64,
        image: &Mat,
        record: bool,
    ) -> Result<Observation> {
        let observation = self.detector.detect(frame_number, image)?;
        if record {
            self.recorder.add_observation(observation.clone());
        }
        Ok(observation)
    }

    /// Annotate an image with an observation
    pub fn annotate_image(
        &mut self,
        image: &Mat,
        observation: &Observation,
    ) -> Result<Mat> {
        self.annotator.annotate(image, observation)
    }

    /// Run webcam demo loop
    pub fn demo(&mut self) -> Result<()> { ... }
}
```

Dynamic dispatch via `Box<dyn Trait>` is chosen over generics (`Tracker<D: Detect, A: Annotate>`) because the demo must switch trackers at runtime (keyboard `c`/`m`/`b` to change tracker type). The vtable cost (~2 CPU cycles) is invisible next to OpenCV inference.

---

## Charuco tracker mapping

### CharucoBoardDefinition (`trackers/charuco/board.rs`)

Rust struct replacing the Python Pydantic `CharucoBoardDefinition`. Validation lives in the constructor — invalid boards cannot be constructed.

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct CharucoBoardDefinition {
    pub squares_x: u32,
    pub squares_y: u32,
    pub square_length_millimeters: f64,
    #[serde(default = "default_marker_length_ratio")]
    pub marker_length_ratio: f64,          // 0.8
    #[serde(default = "default_aruco_dictionary_id")]
    pub aruco_dictionary_id: i32,          // DICT_4X4_250
}
```

Constructor validates:
- `marker_length_ratio * square_length_millimeters < square_length_millimeters`
- `squares_x >= 2 && squares_y >= 2`

Computed methods (replacing Pydantic `@computed_field`):
- `aruco_marker_length_millimeters() -> f64`
- `number_of_corners() -> usize`
- `corner_positions_board_frame() -> Vec<[f64; 3]>`

### CharucoDetector (`trackers/charuco/detector.rs`)

```rust
pub struct CharucoDetector {
    configuration: CharucoDetectorConfig,
    board: opencv::aruco::CharucoBoard,
    detector: opencv::aruco::CharucoDetector,
    aruco_marker_ids: Vec<i32>,
    all_charuco_ids: Vec<i32>,
}
```

`impl Detect for CharucoDetector`:
1. Convert image to grayscale (`COLOR_BGR2GRAY`) if multi-channel
2. Call `detector.detect_board()` — returns charuco corners + ids + aruco marker corners + ids
3. Filter Aruco markers to only those belonging to the board definition (same as Python `valid_indices` check)
4. Construct `Observation` with `ObservationPayload::Charuco { ... }`
5. Build `PointCloud` with one row per charuco corner ID, NaN for undetected corners, visibility=1.0 for detected

### CharucoAnnotator (`trackers/charuco/annotator.rs`)

```rust
pub struct CharucoAnnotator {
    configuration: CharucoAnnotatorConfig,
    observation_history: VecDeque<Observation>,
}
```

`impl Annotate for CharucoAnnotator`:
1. Appends observation to history deque, trims to `show_tracks` length
2. Iterates history in reverse (most recent first), drawing fading corner markers with decreasing opacity/scale
3. On first (current) observation: draws corner ID labels, Aruco marker bounding boxes
4. Lists undetected corner IDs in a sidebar column

Configurable parameters (matching Python `CharucoAnnotatorConfig`):
- `show_tracks: Option<usize>` (default 15), `corner_marker_type`, `corner_marker_size`, `corner_marker_thickness`
- `corner_marker_color: (u8, u8, u8)`, `aruco_lines_thickness`, `aruco_lines_color`

---

## I/O layer

### WebcamCapture (`io/webcam.rs`)

Uses `opencv::videoio::VideoCapture` with `CAP_DSHOW` backend on Windows.

```rust
pub struct WebcamCapture {
    camera: opencv::videoio::VideoCapture,
    frame_width: i32,
    frame_height: i32,
}
```

- `open(camera_index: i32) -> Result<Self>` — opens camera, sets 1280×720, manual exposure
- `read_frame(&mut self) -> Result<Option<(u64, Mat)>>` — grabs + retrieves, returns frame number + image

Uses the `opencv` crate rather than a pure-Rust camera crate (`nokhwa`) so frames are already `opencv::core::Mat` — no pixel format conversion needed before passing to the detector or annotator.

### Recorder (`io/recorder.rs`)

```rust
pub struct Recorder {
    observations: Vec<Observation>,
}
```

`impl Record for Recorder`:
- `add_observation(&mut self, observation: Observation)`
- `clear(&mut self)`
- `observation_count(&self) -> usize`

Additional methods for serialization:
- `to_stacked_array(&self) -> ndarray::Array3<f64>` — all PointClouds stacked into (frames, points, 3)
- `to_json_string(&self) -> Result<String>` — serializes all observations to JSON via `serde_json`
- `save_npy(&self, path: &Path) -> Result<()>` — saves `.npy` via the `npyz` crate

---

## Demo binary (`src/bin/demo.rs`)

Main loop:
1. Open webcam via `WebcamCapture::open(1)`
2. Create `Tracker` with `CharucoDetector` + `CharucoAnnotator` + `Recorder`
3. Per-frame loop:
   - `webcam.read_frame()`
   - `tracker.process_image(frame_number, &frame, record=true)`
   - `tracker.annotate_image(&frame, &observation)`
   - `opencv::highgui::imshow("SkellyTracker Rust", &annotated)`
   - Check keyboard: `q`/ESC to quit
4. On exit: `tracker.recorder.save_npy(...)`

---

## Dependencies

```toml
[dependencies]
opencv = { version = "0.92", features = ["aruco", "videoio", "imgproc", "highgui", "calib3d"] }
serde = { version = "1", features = ["derive"] }
serde_yaml = "0.9"
serde_json = "1"
anyhow = "1"
ndarray = "0.15"
npyz = "0.7"
```

Single OpenCV dependency covers capture (`videoio`), detection (`aruco`, `calib3d` for future `solve_pnp`), annotation (`imgproc`), and display (`highgui`).

---

## Type mapping reference

| Python | Rust |
|--------|------|
| `@dataclass CharucoDetector(BaseDetector)` | `struct CharucoDetector` + `impl Detect` |
| `@dataclass CharucoObservation(BaseObservation)` | `ObservationPayload::Charuco { ... }` |
| `@dataclass CharucoImageAnnotator(BaseImageAnnotator)` | `struct CharucoAnnotator` + `impl Annotate` |
| `CharucoDetectorConfig(BaseDetectorConfig)` | `struct CharucoDetectorConfig` (serde) |
| `CharucoBoardDefinition(BaseModel)` | `struct CharucoBoardDefinition` + validated constructor |
| `PointCloud` | `struct PointCloud` (1:1 mapping) |
| `TrackerType.CHARUCO` (str enum) | `TrackerKind::Charuco` |
| `np.ndarray` | `opencv::core::Mat` (images) / `ndarray::Array2` (point data) |
| `cv2.aruco.CharucoBoard` | `opencv::aruco::CharucoBoard` |
| `cv2.aruco.CharucoDetector` | `opencv::aruco::CharucoDetector` |
| `cv2.solvePnP()` | `opencv::calib3d::solve_pnp()` |
| `cv2.drawMarker()` | `opencv::imgproc::draw_marker()` |
| `cv2.cvtColor(COLOR_BGR2GRAY)` | `opencv::imgproc::cvt_color(COLOR_BGR2GRAY, ...)` |
| Pydantic `@model_validator` | `new() -> Result<Self>` constructor |
| Pydantic `@computed_field` | plain method (no caching needed for arithmetic) |
| ABC inheritance | trait implementation |
| `list[BaseObservation]` | `Vec<Observation>` |

---

## Future tracker additions (design impact)

Adding a new tracker (e.g., MediaPipe) requires:

1. New directory: `src/trackers/mediapipe/` with `detector.rs`, `annotator.rs`, `config.rs`, `observation.rs`, `mediapipe_holistic.yaml`
2. `impl Detect for MediaPipeCompositeDetector` and `impl Annotate for MediaPipeCompositeAnnotator`
3. New variants: `TrackerKind::MediaPipe`, `ObservationPayload::MediaPipe { ... }`
4. One new `match` arm in any code that inspects `ObservationPayload` (e.g., serialization)

**Zero changes required to:** `Tracker`, `Detect`/`Annotate` traits, `Recorder`, `WebcamCapture`, the demo loop.

The YAML composition system (`composed_of`) needed for MediaPipe Holistic (body + left_hand + right_hand + face_contour) will be added to `TrackedObjectDefinition::from_yaml` at that time.
