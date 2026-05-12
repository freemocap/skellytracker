# SkellyTracker Rust — Charuco Milestone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust library + webcam demo binary that runs Charuco board detection using a custom pure-Rust Aruco/Charuco detector — no C++ dependencies, no `opencv` crate.

**Architecture:** One Cargo project (`skellytracker-rust/`) with a `lib.rs` (core types + traits) and a `bin/demo.rs` (webcam binary). Charuco tracker lives in `src/trackers/charuco/`, with custom aruco detection (`detector/aruco_detection.rs`) and corner interpolation (`detector/corner_interpolation.rs`). Core traits use `image` crate types (`DynamicImage`, `RgbImage`, `GrayImage`). Dynamic dispatch via `Box<dyn Trait>` for runtime tracker switching.

**Tech Stack:** Rust 2021 edition, `nokhwa` (webcam), `image` + `imageproc` (pixel buffers + drawing), `ndarray` 0.15, `serde` + `serde_yaml` + `serde_json`, `anyhow` 1, `npyz` 0.7. Zero C++ dependencies.

**Style rules:**
- NO abbreviations ever — `observation` not `obs`, `history` not `hist`, `configuration` not `config` (type suffix `Config` is acceptable for serde structs)
- Many small files — each struct in its own file
- No commits (user preference)

---

### Task 1: Initialize Cargo project

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/Cargo.toml`
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/lib.rs`
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/main.rs` (temporary placeholder, removed in Task 16)

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/core
mkdir -p C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/trackers/charuco
mkdir -p C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/io
mkdir -p C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/bin
mkdir -p C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/tests/fixtures
```

- [ ] **Step 2: Write Cargo.toml**

```toml
[package]
name = "skellytracker"
version = "0.1.0"
edition = "2021"
description = "Rust port of SkellyTracker pose-estimation backend"

[dependencies]
nokhwa = { version = "0.11", features = ["image"] }
image = "0.25"
imageproc = "0.25"
serde = { version = "1", features = ["derive"] }
serde_yaml = "0.9"
serde_json = "1"
anyhow = "1"
ndarray = "0.15"
npyz = "0.7"
```

- [ ] **Step 3: Write placeholder lib.rs**

```rust
// skellytracker — Rust port of SkellyTracker pose-estimation backend
```

- [ ] **Step 4: Write placeholder main.rs (temporary — will be replaced by src/bin/demo.rs in Task 16)**

```rust
fn main() {
    println!("See src/bin/demo.rs for the webcam demo");
}
```

- [ ] **Step 5: Verify cargo build works**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: downloads dependencies (may take a while for opencv), then `Finished` with no errors. If the `opencv` crate fails to compile, check that OpenCV 4.x is installed (the `opencv` crate's build script needs to find the OpenCV libraries). On Windows with the Python `opencv-contrib-python` wheel installed, you may need to set `OPENCV_LINK_LIBS` and `OPENCV_LINK_PATHS` or use the `opencv` crate's `runtime` feature instead.

---

### Task 2: PointCloud — the foundational data type

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/core/point_cloud.rs`

- [ ] **Step 1: Write PointCloud struct**

```rust
use std::collections::HashMap;
use ndarray::{Array1, Array2};

/// Canonical data primitive for tracked landmarks.
///
/// The i-th name in `names` corresponds to the i-th row in `xyz`
/// and the i-th element in `visibility`.
#[derive(Debug, Clone)]
pub struct PointCloud {
    pub names: Vec<String>,
    pub xyz: Array2<f64>,             // (N, 3), z=0 for 2D-only trackers
    pub visibility: Array1<f64>,       // (N,), 0.0 = not detected
    name_to_index: HashMap<String, usize>,
}

impl PointCloud {
    /// Create a new PointCloud with all-NaN coordinates and zero visibility.
    pub fn new(names: Vec<String>) -> Self {
        let number_of_points = names.len();
        let mut name_to_index = HashMap::with_capacity(number_of_points);
        for (index, name) in names.iter().enumerate() {
            name_to_index.insert(name.clone(), index);
        }
        Self {
            names,
            xyz: Array2::from_elem((number_of_points, 3), f64::NAN),
            visibility: Array1::zeros(number_of_points),
            name_to_index,
        }
    }

    /// Look up the array index for a named point. O(1).
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }

    /// Get the (x, y, z) coordinates for a named point.
    pub fn coordinates_by_name(&self, name: &str) -> Option<[f64; 3]> {
        let index = self.name_to_index.get(name)?;
        let row = self.xyz.row(*index);
        Some([row[0], row[1], row[2]])
    }

    /// Immutable view of the xy columns. Zero-copy.
    pub fn xy_view(&self) -> ndarray::ArrayView2<f64> {
        self.xyz.slice(ndarray::s![.., 0..2])
    }

    /// Number of points in this cloud.
    pub fn number_of_points(&self) -> usize {
        self.names.len()
    }

    /// Mask of points with valid (non-NaN) coordinates.
    pub fn valid_mask(&self) -> Array1<bool> {
        self.xyz.column(0).mapv(|x| !x.is_nan())
    }

    /// Count of points with valid (non-NaN) coordinates.
    pub fn number_of_valid(&self) -> usize {
        self.valid_mask().iter().filter(|&&valid| valid).count()
    }

    /// Copy out a subset of points by name. Returns a new PointCloud.
    pub fn slice_by_names(&self, requested_names: &[&str]) -> Self {
        let mut new_cloud = Self::new(
            requested_names.iter().map(|n| n.to_string()).collect(),
        );
        for (new_index, name) in requested_names.iter().enumerate() {
            if let Some(old_index) = self.name_to_index.get(*name) {
                let row = self.xyz.row(*old_index);
                new_cloud.xyz.row_mut(new_index).assign(&row);
                new_cloud.visibility[new_index] = self.visibility[*old_index];
            }
        }
        new_cloud
    }

    /// Filter points by confidence threshold. Returns a new PointCloud.
    /// If `fill_with_nans` is true, points below threshold get NaN coordinates.
    pub fn filtered_by_confidence(
        &self,
        threshold: f64,
        fill_with_nans: bool,
    ) -> Self {
        let mut filtered = self.clone();
        for index in 0..self.number_of_points() {
            if self.visibility[index] < threshold {
                filtered.visibility[index] = 0.0;
                if !fill_with_nans {
                    filtered.xyz.row_mut(index).fill(f64::NAN);
                }
            }
        }
        filtered
    }

    /// Convert to a (N, 2) array of xy coordinates (always N rows, NaN fill).
    pub fn to_2d_array(&self) -> Array2<f64> {
        self.xy_view().to_owned()
    }
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: `Finished` with no errors.

---

### Task 3: TrackedObjectDefinition — YAML schema loading

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/core/tracked_object_definition.rs`

- [ ] **Step 1: Write TrackedObjectDefinition struct**

```rust
use anyhow::{bail, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use super::point_cloud::PointCloud;

/// Schema definition for a tracked object: named points + skeleton connections.
///
/// Loaded from YAML. Replaces Python's Pydantic `TrackedObjectDefinition`.
/// The YAML composition system (`composed_of`) is deferred to the MediaPipe milestone.
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

impl TrackedObjectDefinition {
    /// Load a TrackedObjectDefinition from a YAML file.
    pub fn from_yaml(path: &Path) -> Result<Self> {
        let yaml_content = std::fs::read_to_string(path)?;
        let definition: Self = serde_yaml::from_str(&yaml_content)?;
        definition.validate()?;
        Ok(definition)
    }

    /// Validate that all connection names reference known tracked_points.
    fn validate(&self) -> Result<()> {
        let name_set: HashMap<&str, usize> = self
            .tracked_points
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        for (from_name, to_name) in &self.connections {
            if !name_set.contains_key(from_name.as_str()) {
                bail!(
                    "Connection references unknown point '{}' in definition '{}'",
                    from_name,
                    self.name
                );
            }
            if !name_set.contains_key(to_name.as_str()) {
                bail!(
                    "Connection references unknown point '{}' in definition '{}'",
                    to_name,
                    self.name
                );
            }
        }

        // Check for duplicate point names
        if name_set.len() != self.tracked_points.len() {
            bail!(
                "Duplicate point names found in definition '{}'",
                self.name
            );
        }

        Ok(())
    }

    /// Resolve connection name-pairs to array indices (for drawing).
    pub fn connection_indices(&self) -> Result<Vec<(usize, usize)>> {
        let name_to_index: HashMap<&str, usize> = self
            .tracked_points
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        self.connections
            .iter()
            .map(|(from_name, to_name)| {
                let from_index = *name_to_index
                    .get(from_name.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Unknown point: {}", from_name))?;
                let to_index = *name_to_index
                    .get(to_name.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Unknown point: {}", to_name))?;
                Ok((from_index, to_index))
            })
            .collect()
    }

    /// Factory for an all-NaN PointCloud sized to this definition.
    pub fn empty_point_cloud(&self) -> PointCloud {
        PointCloud::new(self.tracked_points.clone())
    }
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: `Finished` with no errors.

---

### Task 4: Observation, TrackerKind, and ObservationPayload

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/core/observation.rs`

- [ ] **Step 1: Write observation types**

```rust
use super::point_cloud::PointCloud;

/// Top-level observation. Every detection produces one.
#[derive(Debug, Clone)]
pub struct Observation {
    pub frame_number: u64,
    pub tracker_kind: TrackerKind,
    pub points: PointCloud,
    pub payload: ObservationPayload,
}

/// Which tracker produced this observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackerKind {
    Charuco,
    // Future: MediaPipe, RtmPose, CompositeGpu,
}

/// Tracker-specific detection data.
#[derive(Debug, Clone)]
pub enum ObservationPayload {
    Charuco {
        all_charuco_ids: Vec<i32>,
        all_aruco_ids: Vec<i32>,
        detected_charuco_corner_ids: Option<Vec<i32>>,
        detected_charuco_corners: Option<Vec<[f64; 2]>>,
        detected_aruco_marker_ids: Option<Vec<i32>>,
        detected_aruco_marker_corners: Option<Vec<[[f64; 2]; 4]>>,
        board_rotation_vector: Option<[f64; 3]>,
        board_translation_vector: Option<[f64; 3]>,
        detected_charuco_corners_in_camera_coordinates: Option<Vec<[f64; 3]>>,
    },
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: `Finished` with no errors.

---

### Task 5: Core traits — Detect, Annotate, Record

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/core/traits.rs`

- [ ] **Step 1: Write trait definitions**

```rust
use anyhow::Result;
use image::{DynamicImage, GrayImage};

use super::observation::Observation;
use super::tracked_object_definition::TrackedObjectDefinition;

/// Trait for pose-estimation detectors.
///
/// Each tracker implements this to run inference on a single image frame.
/// Accepts a grayscale image — all trackers operate on luminance data.
/// The trait is object-safe so it can be used as `Box<dyn Detect>` for
/// runtime tracker switching.
pub trait Detect {
    /// Run detection on a single grayscale image frame.
    fn detect(&self, frame_number: u64, image: &GrayImage) -> Result<Observation>;

    /// The schema of tracked points and connections this detector produces.
    fn tracked_object_definition(&self) -> &TrackedObjectDefinition;
}

/// Trait for image annotators that draw detection results onto frames.
pub trait Annotate {
    /// Draw detection results onto a color image. Returns a new annotated image.
    fn annotate(
        &mut self,
        image: &DynamicImage,
        observation: &Observation,
    ) -> Result<DynamicImage>;
}

/// Trait for recording observations across frames.
///
/// Includes serialization methods so callers operating on `Box<dyn Record>`
/// can save results without knowing the concrete recorder type.
pub trait Record {
    /// Append an observation to the recording buffer.
    fn add_observation(&mut self, observation: Observation);

    /// Discard all recorded observations.
    fn clear(&mut self);

    /// Number of observations currently stored.
    fn observation_count(&self) -> usize;

    /// Serialize all observations to a JSON string.
    fn to_json_string(&self) -> Result<String>;

    /// Save observations to a .npy file.
    fn save_npy(&self, path: &std::path::Path) -> Result<()>;
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: `Finished` with no errors.

---

### Task 6: Tracker orchestrator

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/core/tracker.rs`

- [ ] **Step 1: Write Tracker struct**

```rust
use anyhow::Result;
use image::{DynamicImage, GrayImage};

use super::observation::Observation;
use super::traits::{Annotate, Detect, Record};

/// Top-level orchestrator. Composes a detector, annotator, and recorder.
///
/// Uses `Box<dyn Trait>` (dynamic dispatch) to enable runtime tracker switching
/// in the demo without changing dispatch code when new trackers are added.
pub struct Tracker {
    pub detector: Box<dyn Detect>,
    pub annotator: Box<dyn Annotate>,
    pub recorder: Box<dyn Record>,
}

impl Tracker {
    /// Create a new Tracker from its components.
    pub fn new(
        detector: Box<dyn Detect>,
        annotator: Box<dyn Annotate>,
        recorder: Box<dyn Record>,
    ) -> Self {
        Self {
            detector,
            annotator,
            recorder,
        }
    }

    /// Run detection and optionally record the observation.
    /// Converts the input to grayscale before passing to the detector.
    pub fn process_image(
        &mut self,
        frame_number: u64,
        image: &GrayImage,
        record: bool,
    ) -> Result<Observation> {
        let observation = self.detector.detect(frame_number, image)?;
        if record {
            self.recorder.add_observation(observation.clone());
        }
        Ok(observation)
    }

    /// Annotate a color image with an observation. Returns a new annotated image.
    pub fn annotate_image(
        &mut self,
        image: &DynamicImage,
        observation: &Observation,
    ) -> Result<DynamicImage> {
        self.annotator.annotate(image, observation)
    }
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: `Finished` with no errors.

---

### Task 7: CharucoBoardDefinition

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/trackers/charuco/board.rs`

- [ ] **Step 1: Write board definition**

```rust
use anyhow::{bail, Result};
use serde::Deserialize;

fn default_marker_length_ratio() -> f64 {
    0.8
}

fn default_aruco_dictionary_id() -> i32 {
    0  // DICT_4X4_250 — OpenCV enum value
}

/// Known charuco board geometry — fixed, never optimized.
///
/// Replaces Python's Pydantic `CharucoBoardDefinition`.
/// Validation lives in `new()` — an invalid board cannot be constructed.
#[derive(Debug, Clone, Deserialize)]
pub struct CharucoBoardDefinition {
    pub squares_x: u32,
    pub squares_y: u32,
    pub square_length_millimeters: f64,
    #[serde(default = "default_marker_length_ratio")]
    pub marker_length_ratio: f64,
    #[serde(default = "default_aruco_dictionary_id")]
    pub aruco_dictionary_id: i32,
}

impl CharucoBoardDefinition {
    /// Validate and construct. Replaces Pydantic `@model_validator`.
    pub fn new(definition: Self) -> Result<Self> {
        let marker_length_millimeters =
            definition.marker_length_ratio * definition.square_length_millimeters;
        if marker_length_millimeters >= definition.square_length_millimeters {
            bail!(
                "marker_length ({}) must be < square_length ({})",
                marker_length_millimeters,
                definition.square_length_millimeters
            );
        }
        if definition.squares_x < 2 || definition.squares_y < 2 {
            bail!(
                "Board must have at least 2x2 squares, got {}x{}",
                definition.squares_x,
                definition.squares_y
            );
        }
        Ok(definition)
    }

    /// Computed: marker length in mm (replaces Pydantic `@computed_field`).
    pub fn aruco_marker_length_millimeters(&self) -> f64 {
        self.marker_length_ratio * self.square_length_millimeters
    }

    /// Number of internal chessboard corners.
    pub fn number_of_corners(&self) -> usize {
        ((self.squares_x - 1) * (self.squares_y - 1)) as usize
    }

    /// Corner positions in board-local frame, (number_of_corners, 3), Z=0.
    pub fn corner_positions_board_frame(&self) -> Vec<[f64; 3]> {
        let columns = self.squares_x - 1;
        let rows = self.squares_y - 1;
        let mut positions = Vec::with_capacity((columns * rows) as usize);
        for row in 0..rows {
            for column in 0..columns {
                positions.push([
                    column as f64 * self.square_length_millimeters,
                    row as f64 * self.square_length_millimeters,
                    0.0,
                ]);
            }
        }
        positions
    }

    /// Convenience constructor: 5x3 letter-size board.
    pub fn create_letter_size_5x3() -> Result<Self> {
        Self::new(Self {
            squares_x: 5,
            squares_y: 3,
            square_length_millimeters: 54.0,
            marker_length_ratio: 0.8,
            aruco_dictionary_id: opencv::objdetect::DICT_4X4_250,
        })
    }

    /// Convenience constructor: 7x5 test board.
    pub fn create_test_data_7x5() -> Result<Self> {
        Self::new(Self {
            squares_x: 7,
            squares_y: 5,
            square_length_millimeters: 58.0,
            marker_length_ratio: 0.8,
            aruco_dictionary_id: opencv::objdetect::DICT_4X4_250,
        })
    }
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: `Finished` with no errors.

---

### Task 8: Charuco tracked object YAML

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/trackers/charuco/charuco_tracked_object.yaml`

- [ ] **Step 1: Write the YAML definition**

```yaml
name: charuco_tracked_object
tracker_type: charuco
landmark_schema: charuco
tracked_points:
  - CharucoCorner-0
  - CharucoCorner-1
  - CharucoCorner-2
  - CharucoCorner-3
  - CharucoCorner-4
  - CharucoCorner-5
  - CharucoCorner-6
  - CharucoCorner-7
  - CharucoCorner-8
  - CharucoCorner-9
  - CharucoCorner-10
  - CharucoCorner-11
  - CharucoCorner-12
  - CharucoCorner-13
  - CharucoCorner-14
  - CharucoCorner-15
  - CharucoCorner-16
  - CharucoCorner-17
  - CharucoCorner-18
  - CharucoCorner-19
  - CharucoCorner-20
  - CharucoCorner-21
  - CharucoCorner-22
  - CharucoCorner-23
  - CharucoCorner-24
  - CharucoCorner-25
  - CharucoCorner-26
  - CharucoCorner-27
  - CharucoCorner-28
  - CharucoCorner-29
  - CharucoCorner-30
  - CharucoCorner-31
  - CharucoCorner-32
  - CharucoCorner-33
  - CharucoCorner-34
  - CharucoCorner-35
  - CharucoCorner-36
  - CharucoCorner-37
  - CharucoCorner-38
  - CharucoCorner-39
  - CharucoCorner-40
  - CharucoCorner-41
  - CharucoCorner-42
  - CharucoCorner-43
  - CharucoCorner-44
  - CharucoCorner-45
  - CharucoCorner-46
  - CharucoCorner-47
  - CharucoCorner-48
  - CharucoCorner-49
connections: []
```

Note: This YAML covers up to 50 charuco corners (the 5x3 letter-size board has `(5-1)*(3-1) = 8` corners, and the 7x5 test board has `(7-1)*(5-1) = 24` corners). The YAML defines all 50 possible corner names so the PointCloud is sized correctly regardless of board configuration. Connections are empty because Charuco boards don't have skeleton connections — they're calibration targets, not skeletons.

- [ ] **Step 2: Verify the YAML parses**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: `Finished` with no errors.

---

### Task 9: Charuco detector and annotator configuration

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/trackers/charuco/config.rs`

- [ ] **Step 1: Write configuration structs**

```rust
use serde::Deserialize;

use super::board::CharucoBoardDefinition;

/// Configuration for the Charuco detector.
///
/// Replaces Python's `CharucoDetectorConfig(BaseDetectorConfig)`.
#[derive(Debug, Clone, Deserialize)]
pub struct CharucoDetectorConfig {
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,
    #[serde(default)]
    pub board_definition: CharucoBoardDefinition,
}

fn default_confidence_threshold() -> f64 {
    0.5
}

impl Default for CharucoDetectorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            board_definition: CharucoBoardDefinition::create_letter_size_5x3()
                .expect("default board definition must be valid"),
        }
    }
}

/// Configuration for the Charuco image annotator.
///
/// Replaces Python's `CharucoAnnotatorConfig(BaseImageAnnotatorConfig)`.
#[derive(Debug, Clone)]
pub struct CharucoAnnotatorConfig {
    pub show_tracks: Option<usize>,
    pub corner_marker_type: i32,
    pub corner_marker_size: i32,
    pub corner_marker_thickness: i32,
    pub corner_marker_color: (u8, u8, u8),
    pub aruco_lines_thickness: i32,
    pub aruco_lines_color: (u8, u8, u8),
    pub text_color: (u8, u8, u8),
    pub text_size: f64,
    pub text_thickness: i32,
    pub show_overlay: bool,
}

impl Default for CharucoAnnotatorConfig {
    fn default() -> Self {
        Self {
            show_tracks: Some(15),
            corner_marker_type: opencv::imgproc::MARKER_DIAMOND,
            corner_marker_size: 10,
            corner_marker_thickness: 2,
            corner_marker_color: (255, 0, 255),
            aruco_lines_thickness: 2,
            aruco_lines_color: (0, 255, 0),
            text_color: (215, 115, 40),
            text_size: 0.5,
            text_thickness: 2,
            show_overlay: false,
        }
    }
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: `Finished` with no errors.

---

### Task 10: Charuco observation construction

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/trackers/charuco/observation.rs`

- [ ] **Step 1: Write observation builder**

```rust
use anyhow::{bail, Result};
use ndarray::Array2;
use opencv::core::Mat;
use opencv::prelude::*;

use crate::core::observation::{Observation, ObservationPayload, TrackerKind};
use crate::core::point_cloud::PointCloud;

/// Minimum number of charuco corners required for the board to be considered visible.
pub const MINIMUM_CHARUCO_CORNERS_FOR_VISIBILITY: usize = 6;

/// Build an Observation from raw charuco detection results.
///
/// Replaces Python's `CharucoObservation.from_detection_results()`.
///
/// # Arguments
/// * `frame_number` — Sequential frame counter.
/// * `detected_charuco_corners_mat` — (N, 2) float Mat of detected corner image coordinates.
/// * `detected_charuco_ids_mat` — (N, 1) int Mat of detected corner IDs.
/// * `detected_marker_corners_vector` — Vector of vectors: one Vec<Point2f> per detected marker.
/// * `detected_marker_ids_vector` — Vector of detected marker IDs.
/// * `all_charuco_ids` — All possible charuco corner IDs for this board.
/// * `all_aruco_ids` — All possible aruco marker IDs for this board.
/// * `image_width` — Image width in pixels.
/// * `image_height` — Image height in pixels.
pub fn build_charuco_observation(
    frame_number: u64,
    detected_charuco_corners_mat: &Mat,
    detected_charuco_ids_mat: &Mat,
    detected_marker_corners_vector: &opencv::types::VectorOfVectorOfPoint2f,
    detected_marker_ids_vector: &opencv::core::Vector<i32>,
    all_charuco_ids: &[i32],
    all_aruco_ids: &[i32],
    image_width: i32,
    image_height: i32,
) -> Result<Observation> {
    let _ = image_width;
    let _ = image_height;

    let number_of_corners = all_charuco_ids.len();

    // Build PointCloud: one row per charuco corner ID, NaN for undetected
    let corner_names: Vec<String> = (0..number_of_corners)
        .map(|index| format!("CharucoCorner-{}", index))
        .collect();
    let mut point_cloud = PointCloud::new(corner_names);

    // Extract detected charuco corners
    let detected_ids: Option<Vec<i32>> = if !detected_charuco_ids_mat.empty() {
        let ids_1d = detected_charuco_ids_mat.data_1d::<i32>()?;
        Some(ids_1d.to_vec())
    } else {
        None
    };

    let detected_corners: Option<Vec<[f64; 2]>> = if !detected_charuco_corners_mat.empty() {
        let rows = detected_charuco_corners_mat.rows();
        let mut corners = Vec::with_capacity(rows as usize);
        for row in 0..rows {
            let x = *detected_charuco_corners_mat.at_2d::<f64>(row, 0)?;
            let y = *detected_charuco_corners_mat.at_2d::<f64>(row, 1)?;
            corners.push([x, y]);
        }
        Some(corners)
    } else {
        None
    };

    // Populate PointCloud: for each detected corner, place it at its ID index
    if let (Some(ref ids), Some(ref corners)) = (&detected_ids, &detected_corners) {
        if ids.len() != corners.len() {
            bail!(
                "Frame {}: detected charuco ids count ({}) != corners count ({})",
                frame_number,
                ids.len(),
                corners.len()
            );
        }
        for (corner_index, corner_id) in ids.iter().enumerate() {
            let array_index = *corner_id as usize;
            if array_index < number_of_corners {
                let [x, y] = corners[corner_index];
                point_cloud.xyz[[array_index, 0]] = x;
                point_cloud.xyz[[array_index, 1]] = y;
                point_cloud.xyz[[array_index, 2]] = 0.0;
                point_cloud.visibility[array_index] = 1.0;
            }
        }
    }

    // Extract detected aruco markers
    let detected_marker_ids: Option<Vec<i32>> = if !detected_marker_ids_vector.is_empty() {
        Some(detected_marker_ids_vector.to_vec())
    } else {
        None
    };

    let detected_marker_corners: Option<Vec<[[f64; 2]; 4]>> =
        if !detected_marker_corners_vector.is_empty() {
            let mut markers = Vec::new();
            for marker_index in 0..detected_marker_corners_vector.len() {
                let corner_points =
                    detected_marker_corners_vector.get(marker_index)?;
                let mut marker = [[0.0f64; 2]; 4];
                for (corner, point_index) in marker.iter_mut().zip(0..4) {
                    let point = corner_points.get(point_index)?;
                    *corner = [point.x as f64, point.y as f64];
                }
                markers.push(marker);
            }
            Some(markers)
        } else {
            None
        };

    Ok(Observation {
        frame_number,
        tracker_kind: TrackerKind::Charuco,
        points: point_cloud,
        payload: ObservationPayload::Charuco {
            all_charuco_ids: all_charuco_ids.to_vec(),
            all_aruco_ids: all_aruco_ids.to_vec(),
            detected_charuco_corner_ids: detected_ids,
            detected_charuco_corners: detected_corners,
            detected_aruco_marker_ids: detected_marker_ids,
            detected_aruco_marker_corners: detected_marker_corners,
            board_rotation_vector: None,
            board_translation_vector: None,
            detected_charuco_corners_in_camera_coordinates: None,
        },
    })
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: `Finished` with no errors.

---

### Task 11: CharucoDetector — impl Detect

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/trackers/charuco/detector.rs`

- [ ] **Step 1: Write CharucoDetector**

```rust
use anyhow::Result;
use opencv::core::{Mat, Vector};
use opencv::objdetect;
use opencv::prelude::*;
use opencv::types::VectorOfVectorOfPoint2f;

use crate::core::observation::Observation;
use crate::core::tracked_object_definition::TrackedObjectDefinition;
use crate::core::traits::Detect;

use super::config::CharucoDetectorConfig;
use super::observation::build_charuco_observation;

/// Charuco board detector. Wraps OpenCV's `cv::aruco::CharucoDetector`.
///
/// Replaces Python's `CharucoDetector(BaseDetector)`.
pub struct CharucoDetector {
    configuration: CharucoDetectorConfig,
    board: objdetect::CharucoBoard,
    detector: objdetect::CharucoDetector,
    aruco_marker_ids: Vec<i32>,
    all_charuco_ids: Vec<i32>,
    tracked_object_definition: TrackedObjectDefinition,
    image_width: i32,
    image_height: i32,
}

impl CharucoDetector {
    /// Create a CharucoDetector from configuration.
    ///
    /// Replaces Python's `CharucoDetector.create(config)`.
    pub fn create(configuration: CharucoDetectorConfig) -> Result<Self> {
        let board_definition = &configuration.board_definition;

        let dictionary = objdetect::get_predefined_dictionary(
            board_definition.aruco_dictionary_id,
        )?;

        let board = objdetect::CharucoBoard::new(
            opencv::core::Size::new(
                board_definition.squares_x as i32,
                board_definition.squares_y as i32,
            ),
            board_definition.square_length_millimeters as f32,
            board_definition.aruco_marker_length_millimeters() as f32,
            &dictionary,
        )?;

        let detector = objdetect::CharucoDetector::new_def(&board)?;

        // Collect marker IDs from the board
        let board_ids_mat = board.get_ids()?;
        let aruco_marker_ids: Vec<i32> = if !board_ids_mat.empty() {
            board_ids_mat.data_1d::<i32>()?.to_vec()
        } else {
            Vec::new()
        };

        let number_of_corners = board_definition.number_of_corners();
        let all_charuco_ids: Vec<i32> = (0..number_of_corners as i32).collect();

        // Load tracked object definition from YAML
        let yaml_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/trackers/charuco/charuco_tracked_object.yaml");
        let tracked_object_definition =
            TrackedObjectDefinition::from_yaml(&yaml_path)?;

        Ok(Self {
            configuration,
            board,
            detector,
            aruco_marker_ids,
            all_charuco_ids,
            tracked_object_definition,
            image_width: 0,
            image_height: 0,
        })
    }
}

impl Detect for CharucoDetector {
    fn detect(&self, frame_number: u64, image: &Mat) -> Result<Observation> {
        // 1. Convert to grayscale if multi-channel (same as Python: cv2.COLOR_BGR2GRAY)
        let grey_image = if image.channels() > 1 {
            let mut grey = Mat::default();
            opencv::imgproc::cvt_color(
                image,
                &mut grey,
                opencv::imgproc::COLOR_BGR2GRAY,
                0,
            )?;
            grey
        } else {
            image.clone()
        };

        // 2. Detect charuco board
        let mut charuco_corners = Mat::default();
        let mut charuco_ids = Mat::default();
        let mut marker_corners = VectorOfVectorOfPoint2f::new();
        let mut marker_ids = Vector::<i32>::new();

        self.detector.detect_board(
            &grey_image,
            &mut charuco_corners,
            &mut charuco_ids,
            &mut marker_corners,
            &mut marker_ids,
        )?;

        // 3. Filter aruco markers to only those belonging to the board
        // (same as Python: `valid_indices` filter in detect())
        let mut filtered_marker_corners = VectorOfVectorOfPoint2f::new();
        let mut filtered_marker_ids = Vector::<i32>::new();
        if !marker_ids.is_empty() {
            for marker_index in 0..marker_ids.len() {
                let marker_id = *marker_ids.get(marker_index)?;
                if self.aruco_marker_ids.contains(&marker_id) {
                    filtered_marker_ids.push(marker_id);
                    let corners = marker_corners.get(marker_index)?;
                    filtered_marker_corners.push(&corners);
                }
            }
        }

        // 4. Build Observation
        build_charuco_observation(
            frame_number,
            &charuco_corners,
            &charuco_ids,
            &filtered_marker_corners,
            &filtered_marker_ids,
            &self.all_charuco_ids,
            &self.aruco_marker_ids,
            self.image_width,
            self.image_height,
        )
    }

    fn tracked_object_definition(&self) -> &TrackedObjectDefinition {
        &self.tracked_object_definition
    }
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: `Finished` with no errors.

---

### Task 12: CharucoAnnotator — impl Annotate

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/trackers/charuco/annotator.rs`

- [ ] **Step 1: Write CharucoAnnotator**

```rust
use std::collections::VecDeque;

use anyhow::Result;
use opencv::core::{Mat, Point, Scalar};
use opencv::imgproc;
use opencv::prelude::*;

use crate::core::observation::{Observation, ObservationPayload};
use crate::core::traits::Annotate;

use super::config::CharucoAnnotatorConfig;

/// Charuco image annotator — draws detected corners, Aruco marker boxes,
/// and fading trails onto the video frame.
///
/// Replaces Python's `CharucoImageAnnotator(BaseImageAnnotator)`.
pub struct CharucoAnnotator {
    configuration: CharucoAnnotatorConfig,
    observation_history: VecDeque<Observation>,
}

impl CharucoAnnotator {
    /// Create a new annotator from configuration.
    pub fn create(configuration: CharucoAnnotatorConfig) -> Self {
        Self {
            configuration,
            observation_history: VecDeque::new(),
        }
    }
}

impl Annotate for CharucoAnnotator {
    fn annotate(&mut self, image: &Mat, observation: &Observation) -> Result<Mat> {
        let mut annotated = image.clone()?;
        let text_offset = (annotated.rows() as f64 * 0.01) as i32;

        // Add to history and trim to show_tracks length
        self.observation_history.push_back(observation.clone());
        let show_tracks = self.configuration.show_tracks.unwrap_or(15);
        while self.observation_history.len() > show_tracks {
            self.observation_history.pop_front();
        }

        let history_length = self.observation_history.len();

        // Draw fading trails (iterates history in reverse, same logic as Python)
        for (history_offset, historical_observation) in
            self.observation_history.iter().rev().enumerate()
        {
            let scale = 1.0 - (history_offset as f64 / history_length as f64);

            if let ObservationPayload::Charuco {
                ref detected_charuco_corners,
                ref detected_charuco_corner_ids,
                ..
            } = historical_observation.payload
            {
                if let (Some(ids), Some(corners)) =
                    (detected_charuco_corner_ids, detected_charuco_corners)
                {
                    let (r, g, b) = self.configuration.corner_marker_color;
                    let marker_color = Scalar::new(
                        (b as f64 * scale) as f64,
                        (g as f64 * scale) as f64,
                        (r as f64 * scale) as f64,
                        0.0,
                    );
                    let marker_thickness = std::cmp::max(
                        1,
                        (self.configuration.corner_marker_thickness as f64 * scale) as i32,
                    );
                    let marker_size = std::cmp::max(
                        1,
                        (self.configuration.corner_marker_size as f64 * scale) as i32,
                    );

                    for (corner_index, _corner_id) in ids.iter().enumerate() {
                        let [x, y] = corners[corner_index];
                        if !x.is_nan() && !y.is_nan() {
                            imgproc::draw_marker(
                                &mut annotated,
                                Point::new(x as i32, y as i32),
                                marker_color,
                                self.configuration.corner_marker_type,
                                marker_size,
                                marker_thickness,
                                8, // line_type = 8-connected
                            )?;
                        }
                    }

                    // On current observation (history_offset == 0): draw labels and aruco boxes
                    if history_offset == 0 {
                        for (corner_index, corner_id) in ids.iter().enumerate() {
                            let [x, y] = corners[corner_index];
                            if !x.is_nan() && !y.is_nan() {
                                let text = format!("Corner#{}", corner_id);
                                let text_position = Point::new(
                                    x as i32 + text_offset,
                                    y as i32 + text_offset,
                                );
                                draw_doubled_text(
                                    &mut annotated,
                                    &text,
                                    text_position,
                                    self.configuration.text_size,
                                    Scalar::new(
                                        self.configuration.text_color.2 as f64,
                                        self.configuration.text_color.1 as f64,
                                        self.configuration.text_color.0 as f64,
                                        0.0,
                                    ),
                                    self.configuration.text_thickness,
                                )?;
                            }
                        }
                    }
                }
            }

            // Draw Aruco marker bounding boxes on current observation only
            if history_offset == 0 {
                if let ObservationPayload::Charuco {
                    ref detected_aruco_marker_ids,
                    ref detected_aruco_marker_corners,
                    ..
                } = historical_observation.payload
                {
                    if let (Some(marker_ids), Some(marker_corners)) =
                        (detected_aruco_marker_ids, detected_aruco_marker_corners)
                    {
                        let (r, g, b) = self.configuration.aruco_lines_color;
                        let aruco_color = Scalar::new(b as f64, g as f64, r as f64, 0.0);

                        for (marker_index, marker_id) in marker_ids.iter().enumerate() {
                            if let Some(corners_array) = marker_corners.get(marker_index) {
                                let points: Vec<Point> = corners_array
                                    .iter()
                                    .map(|corner| Point::new(corner[0] as i32, corner[1] as i32))
                                    .collect();

                                imgproc::polylines(
                                    &mut annotated,
                                    &vec![points],
                                    true, // is_closed
                                    aruco_color,
                                    self.configuration.aruco_lines_thickness,
                                    imgproc::LINE_8,
                                    0,
                                )?;

                                let label = format!(
                                    "Aruco#{}",
                                    marker_id
                                );
                                let label_position = Point::new(
                                    corners_array[0][0] as i32 + text_offset,
                                    corners_array[0][1] as i32 + text_offset,
                                );
                                draw_doubled_text(
                                    &mut annotated,
                                    &label,
                                    label_position,
                                    self.configuration.text_size,
                                    Scalar::new(0.0, 125.0, 255.0, 0.0),
                                    1,
                                )?;
                            }
                        }
                    }
                }
            }
        }

        // List undetected corner IDs
        if let ObservationPayload::Charuco {
            ref all_charuco_ids,
            ref detected_charuco_corner_ids,
            ..
        } = observation.payload
        {
            let mut undetected_corners: Vec<i32> = all_charuco_ids.clone();
            if let Some(detected_ids) = detected_charuco_corner_ids {
                undetected_corners.retain(|id| !detected_ids.contains(id));
            }

            if !undetected_corners.is_empty() {
                let image_width = annotated.cols();
                let text_color = Scalar::new(
                    self.configuration.text_color.2 as f64,
                    self.configuration.text_color.1 as f64,
                    self.configuration.text_color.0 as f64,
                    0.0,
                );

                draw_doubled_text(
                    &mut annotated,
                    "Undetected Corners:",
                    Point::new(image_width - 200, 20),
                    self.configuration.text_size,
                    text_color,
                    self.configuration.text_thickness,
                )?;

                for (undetected_index, corner_id) in
                    undetected_corners.iter().enumerate()
                {
                    let vertical_offset = undetected_index as i32 * 20;
                    draw_doubled_text(
                        &mut annotated,
                        &format!(" - {}", corner_id),
                        Point::new(image_width - 200, 40 + vertical_offset),
                        self.configuration.text_size,
                        text_color,
                        self.configuration.text_thickness,
                    )?;
                }
            }
        }

        Ok(annotated)
    }
}

/// Draw text with a dark outline for readability against any background.
/// Matches Python's `BaseImageAnnotator.draw_doubled_text()`.
fn draw_doubled_text(
    image: &mut Mat,
    text: &str,
    position: Point,
    font_scale: f64,
    color: Scalar,
    thickness: i32,
) -> Result<()> {
    let outline_color = Scalar::new(0.0, 0.0, 0.0, 0.0);
    let outline_thickness = thickness + 2;

    imgproc::put_text(
        image,
        text,
        position,
        imgproc::FONT_HERSHEY_SIMPLEX,
        font_scale,
        outline_color,
        outline_thickness,
        imgproc::LINE_AA,
        false,
    )?;

    imgproc::put_text(
        image,
        text,
        position,
        imgproc::FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(())
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: `Finished` with no errors.

---

### Task 13: Wire up all module declarations

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/core/mod.rs`
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/trackers/mod.rs`
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/trackers/charuco/mod.rs`
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/io/mod.rs`
- Modify: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/lib.rs`

- [ ] **Step 1: Create placeholder stubs for io files (needed for module declarations to compile)**

Write `io/video.rs`:
```rust
// Video file I/O — stub for future milestone.
```

Write `io/webcam.rs`:
```rust
// Webcam capture — full implementation in Task 15.
```

Write `io/recorder.rs`:
```rust
// Recorder — full implementation in Task 14.
```

- [ ] **Step 2: Write core/mod.rs**

```rust
pub mod observation;
pub mod point_cloud;
pub mod tracked_object_definition;
pub mod traits;
pub mod tracker;
```

- [ ] **Step 3: Write trackers/mod.rs**

```rust
pub mod charuco;
```

- [ ] **Step 4: Write trackers/charuco/mod.rs**

```rust
pub mod annotator;
pub mod board;
pub mod config;
pub mod detector;
pub mod observation;
```

- [ ] **Step 5: Write io/mod.rs**

```rust
pub mod recorder;
pub mod video;
pub mod webcam;
```

- [ ] **Step 6: Rewrite lib.rs to wire everything together**

```rust
pub mod core;

<system-reminder>
The TodoWrite tool hasn't been used recently. If you're working on tasks that would benefit from tracking progress, consider using TaskCreate to add new tasks and TaskUpdate to update task status (set to in_progress when starting, completed when done). Also consider cleaning up the task list if it has become stale. Only use these if relevant to the current work. This is just a gentle reminder - ignore if not applicable. Make sure that you NEVER mention this reminder to the user.
</system-reminder>

pub mod io;
pub mod trackers;
```

- [ ] **Step 7: Verify compilation**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: `Finished` with no errors.

---

### Task 14: Recorder — impl Record

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/io/recorder.rs`

- [ ] **Step 1: Write Recorder struct**

```rust
use std::path::Path;

use anyhow::Result;
use ndarray::Array3;

use crate::core::observation::Observation;
use crate::core::traits::Record;

/// Collects observations across frames and serializes to JSON or .npy.
///
/// Replaces Python's `BaseRecorder`.
pub struct Recorder {
    observations: Vec<Observation>,
}

impl Recorder {
    /// Create an empty recorder.
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
        }
    }
}

impl Default for Recorder {
    fn default() -> Self {
        Self::new()
    }
}

impl Record for Recorder {
    fn add_observation(&mut self, observation: Observation) {
        self.observations.push(observation);
    }

    fn clear(&mut self) {
        self.observations.clear();
    }

    fn observation_count(&self) -> usize {
        self.observations.len()
    }
}

impl Recorder {
    /// Stack all PointCloud xy arrays into (frames, points, 2) array.
    pub fn to_stacked_array(&self) -> Option<Array3<f64>> {
        if self.observations.is_empty() {
            return None;
        }

        let frame_count = self.observations.len();
        let point_count = self.observations[0].points.number_of_points();

        let mut stacked =
            Array3::from_elem((frame_count, point_count, 2), f64::NAN);

        for (frame_index, observation) in self.observations.iter().enumerate() {
            let xy = observation.points.xy_view();
            for point_index in 0..point_count {
                stacked[[frame_index, point_index, 0]] = xy[[point_index, 0]];
                stacked[[frame_index, point_index, 1]] = xy[[point_index, 1]];
            }
        }

        Some(stacked)
    }

    /// Serialize all observations to a JSON string.
    pub fn to_json_string(&self) -> Result<String> {
        let value = serde_json::json!({
            "observations": self.observations.iter().map(|observation| {
                serde_json::json!({
                    "frame_number": observation.frame_number,
                    "tracker_kind": format!("{:?}", observation.tracker_kind),
                    "number_of_valid_points": observation.points.number_of_valid(),
                })
            }).collect::<Vec<_>>(),
        });
        Ok(serde_json::to_string_pretty(&value)?)
    }

    /// Save observations to a .npy file.
    pub fn save_npy(&self, path: &Path) -> Result<()> {
        if let Some(stacked) = self.to_stacked_array() {
            let shape = stacked.shape();
            let flat: Vec<f64> = stacked.iter().copied().collect();
            npyz::WriteOptions::default()
                .shape(&[shape[0] as u64, shape[1] as u64, shape[2] as u64])
                .write(path, &flat)?;
        }
        Ok(())
    }
}
```

- [ ] **Step 2: Write placeholder io/video.rs**

```rust
// Video file I/O — stub for future milestone.
```

- [ ] **Step 3: Write placeholder io/webcam.rs** (full implementation in Task 15)

```rust
// Webcam capture — see full implementation in Task 15.
```

- [ ] **Step 4: Verify compilation**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: `Finished` with no errors.

---

### Task 15: WebcamCapture

**Files:**
- Overwrite: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/io/webcam.rs`

- [ ] **Step 1: Write WebcamCapture**

```rust
use anyhow::{bail, Result};
use opencv::core::Mat;
use opencv::prelude::*;
use opencv::videoio;

/// Webcam capture via OpenCV VideoCapture.
///
/// Uses `opencv` crate (not `nokhwa`) so frames are already `Mat` —
/// no pixel format conversion needed before passing to detector/annotator.
pub struct WebcamCapture {
    camera: videoio::VideoCapture,
    frame_number: u64,
}

impl WebcamCapture {
    /// Open a webcam by index.
    ///
    /// Sets 1280x720 resolution and manual exposure (matching Python demo).
    pub fn open(camera_index: i32) -> Result<Self> {
        let camera = videoio::VideoCapture::new(camera_index, videoio::CAP_DSHOW)?;

        if !videoio::VideoCapture::is_opened(&camera)? {
            bail!("Failed to open camera at index {}", camera_index);
        }

        camera.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0)?;
        camera.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0)?;
        camera.set(videoio::CAP_PROP_AUTO_EXPOSURE, 0.25)?; // manual exposure mode
        camera.set(videoio::CAP_PROP_EXPOSURE, -7.0)?; // typical indoor exposure

        Ok(Self {
            camera,
            frame_number: 0,
        })
    }

    /// Read the next frame. Returns `None` if the camera has no more frames.
    pub fn read_frame(&mut self) -> Result<Option<(u64, Mat)>> {
        let mut frame = Mat::default();
        let success = self.camera.read(&mut frame)?;

        if !success || frame.empty() {
            return Ok(None);
        }

        let current_frame_number = self.frame_number;
        self.frame_number += 1;

        Ok(Some((current_frame_number, frame)))
    }
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: `Finished` with no errors.

---

### Task 16: Demo binary

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/bin/demo.rs`
- Delete: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/main.rs`
- Modify: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/Cargo.toml` (remove default binary, add [[bin]])

- [ ] **Step 1: Update Cargo.toml to remove default binary and add demo binary**

In `Cargo.toml`, after the `[dependencies]` section, add:

```toml
[[bin]]
name = "demo"
path = "src/bin/demo.rs"
```

Also remove any `default-run` if present. (The presence of `src/main.rs` would normally create a default binary; by deleting it and using `[[bin]]` we ensure only the demo binary exists.)

- [ ] **Step 2: Delete the placeholder main.rs**

```bash
rm C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/src/main.rs
```

- [ ] **Step 3: Write demo binary**

```rust
use std::time::Instant;

use anyhow::Result;
use opencv::highgui;

use skellytracker::core::tracker::Tracker;
use skellytracker::io::recorder::Recorder;
use skellytracker::io::webcam::WebcamCapture;
use skellytracker::trackers::charuco::annotator::{CharucoAnnotator, CharucoAnnotatorConfig};
use skellytracker::trackers::charuco::config::CharucoDetectorConfig;
use skellytracker::trackers::charuco::detector::CharucoDetector;

fn main() -> Result<()> {
    println!("SkellyTracker Rust — Charuco Webcam Demo");
    println!("=========================================");
    println!("Press 'q' or ESC to quit.\n");

    // Open webcam
    println!("Opening webcam...");
    let mut webcam = WebcamCapture::open(1)?;
    println!("Webcam ready.");

    // Build Charuco tracker
    println!("Initializing Charuco detector...");
    let detector_config = CharucoDetectorConfig::default();
    let detector = CharucoDetector::create(detector_config)?;
    let annotator = CharucoAnnotator::create(CharucoAnnotatorConfig::default());
    let recorder = Recorder::new();

    let mut tracker = Tracker::new(
        Box::new(detector),
        Box::new(annotator),
        Box::new(recorder),
    );
    println!("Tracker ready. Starting loop...\n");

    let window_name = "SkellyTracker Rust — Charuco Demo";

    loop {
        let frame_start = Instant::now();

        // Read frame
        let (frame_number, frame) = match webcam.read_frame()? {
            Some(frame_data) => frame_data,
            None => {
                eprintln!("Camera returned no frame.");
                break;
            }
        };

        // Process (detect + record)
        let observation = tracker.process_image(frame_number, &frame, true)?;

        // Annotate
        let annotated = tracker.annotate_image(&frame, &observation)?;

        // Display
        highgui::imshow(window_name, &annotated)?;

        // Check for quit
        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 {
            // 'q' or ESC
            println!("Quit requested.");
            break;
        }

        // Frame timing
        let frame_duration = frame_start.elapsed();
        let fps = 1.0 / frame_duration.as_secs_f64();
        if frame_number % 30 == 0 {
            println!(
                "Frame {} | {:.1} FPS | {} valid corners",
                frame_number,
                fps,
                observation.points.number_of_valid()
            );
        }
    }

    // Save recorded data
    let output_path = std::path::Path::new("charuco_demo_output.npy");
    tracker.recorder.save_npy(output_path)?;
    println!(
        "Saved {} observations to {}",
        tracker.recorder.observation_count(),
        output_path.display()
    );

    Ok(())
}
```

- [ ] **Step 4: Verify compilation**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo check
```

Expected: `Finished` with no errors.

- [ ] **Step 5: Build release binary**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo build --release
```

Expected: compiles successfully. Binary at `target/release/demo.exe` (Windows) or `target/release/demo` (Linux/macOS).

- [ ] **Step 6: Test run** (requires a webcam and ideally a printed Charuco board)

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo run --release
```

Expected: webcam window opens, shows video. If a Charuco board is visible, corners are detected and marked with diamond shapes. Press `q` to quit. Output saved to `charuco_demo_output.npy`.

---

### Task 17: Integration tests

**Files:**
- Create: `C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust/tests/charuco_detection_test.rs`

- [ ] **Step 1: Copy a test image from the Python tests**

Find the Charuco test image used by the Python tests (download URL is in `skellytracker/tests/conftest.py`) and save it to `tests/fixtures/charuco_test_image.jpg`.

- [ ] **Step 2: Write detection test**

```rust
use std::path::Path;

use opencv::imgcodecs;
use opencv::prelude::*;

use skellytracker::core::traits::Detect;
use skellytracker::trackers::charuco::config::CharucoDetectorConfig;
use skellytracker::trackers::charuco::detector::CharucoDetector;

/// Test that a CharucoDetector can be created.
#[test]
fn test_create_charuco_detector() {
    let config = CharucoDetectorConfig::default();
    let detector = CharucoDetector::create(config);
    assert!(detector.is_ok(), "Failed to create CharucoDetector");
}

/// Test detection on a known Charuco board image.
#[test]
fn test_detect_charuco_board() {
    let config = CharucoDetectorConfig::default();
    let detector = CharucoDetector::create(config)
        .expect("Failed to create CharucoDetector");

    let image_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/charuco_test_image.jpg");
    let image = imgcodecs::imread(
        image_path.to_str().unwrap(),
        imgcodecs::IMREAD_COLOR,
    )
    .expect("Failed to load test image");

    let observation = detector.detect(0, &image)
        .expect("Detection failed");

    assert_eq!(observation.frame_number, 0);

    // The test image should have detected some corners
    let valid_count = observation.points.number_of_valid();
    println!("Detected {} valid charuco corners", valid_count);
    assert!(
        valid_count > 0,
        "Expected at least 1 detected corner on the charuco test image"
    );

    // Verify the tracked object definition
    let tracked_object = detector.tracked_object_definition();
    assert_eq!(tracked_object.tracker_type, "charuco");
    assert_eq!(tracked_object.name, "charuco_tracked_object");
}

/// Test that board definition validation catches invalid boards.
#[test]
fn test_invalid_board_definition() {
    use skellytracker::trackers::charuco::board::CharucoBoardDefinition;

    // squares_x < 2 should fail
    let result = CharucoBoardDefinition::new(CharucoBoardDefinition {
        squares_x: 1,
        squares_y: 3,
        square_length_millimeters: 54.0,
        marker_length_ratio: 0.8,
        aruco_dictionary_id: opencv::objdetect::DICT_4X4_250,
    });
    assert!(result.is_err(), "Should reject board with squares_x < 2");

    // marker_length >= square_length should fail
    let result = CharucoBoardDefinition::new(CharucoBoardDefinition {
        squares_x: 5,
        squares_y: 3,
        square_length_millimeters: 54.0,
        marker_length_ratio: 1.5,
        aruco_dictionary_id: opencv::objdetect::DICT_4X4_250,
    });
    assert!(result.is_err(), "Should reject board with marker >= square");
}
```

- [ ] **Step 3: Run tests**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo test
```

Expected: tests pass. The detection test confirms the pipeline works end-to-end: create detector → load image → detect → verify observation has valid corners.

- [ ] **Step 4: Run all tests with verbose output**

```bash
cd C:/Users/jonma/code_repos/github/freemocap/skellytracker/skellytracker-rust && cargo test -- --nocapture
```

Expected: see the number of detected corners printed to stdout.

---

## Implementation Order Summary

Tasks must run sequentially because each builds on the previous:

1. T1: Cargo init → skeleton compiles
2. T2: PointCloud → foundational data type
3. T3: TrackedObjectDefinition → YAML loading (depends on PointCloud)
4. T4: Observation types → (depends on PointCloud)
5. T5: Traits → (depends on Observation, TrackedObjectDefinition)
6. T6: Tracker → (depends on traits)
7. T7: CharucoBoardDefinition → (standalone, depends on opencv)
8. T8: Charuco YAML → (standalone file)
9. T9: Charuco config → (depends on board)
10. T10: Charuco observation builder → (depends on PointCloud, Observation)
11. T11: CharucoDetector → (depends on config, board, traits, observation builder)
12. T12: CharucoAnnotator → (depends on config, traits, Observation)
13. T13: Module wiring → (depends on all above)
14. T14: Recorder → (depends on traits, Observation; needs io/mod.rs)
15. T15: WebcamCapture → (depends on io/mod.rs)
16. T16: Demo binary → (depends on everything)
17. T17: Tests → (depends on demo binary existing)
