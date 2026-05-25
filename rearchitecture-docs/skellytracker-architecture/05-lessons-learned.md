# Lessons Learned

> Mistakes, gotchas, and patterns to reuse. Everything we'd tell ourselves if we were starting the next tracker translation tomorrow.

## RULE #0 — DATA MODEL PARITY IS NON-NEGOTIABLE

**The Rust Observation MUST have every field the Python Observation has, with the same names, same types, and same semantics.** Downstream consumers MUST NOT be able to tell whether an observation came from the Rust or Python backend.

- Match field names exactly (e.g. `detected_charuco_corners_image_coordinates`, not `detected_charuco_corners`)
- Match types (e.g. `Vec<[f64; 2]>` not `Vec<Point2f>` — arrays serialize the same as Python lists)
- Include deferred fields as `None` — they exist but aren't populated yet
- The `to_json()` output must include every field the Python `to_json_string()` would
- Every `#[getter]` / property Python exposes must have a Rust equivalent accessible via JSON

**Verification:** serialize both Python and Rust observations for the same frame to JSON and assert the dicts are identical.

## Mistakes That Cost Hours

### 1. `unwrap_or_default()` on OpenCV calls — DO NOT DO THIS

```rust
// WRONG — silently swallows ALL errors
imgproc::cvt_color(image, &mut gray, COLOR_BGR2GRAY, 0, ALGO_HINT_DEFAULT)
    .unwrap_or_default();
```

**Symptom:** Tracker produces zero detections with no error message. Frame looks fine. No crash. Just silently broken.

**Fix:**
```rust
if imgproc::cvt_color(image, &mut gray, COLOR_BGR2GRAY, 0, ALGO_HINT_DEFAULT).is_err() {
    eprintln!("[skellytracker-rust] cvt_color failed — returning empty observation");
    return empty_observation();
}
```

**Rule:** Every OpenCV call in a detection hot loop gets an `is_err()` check with `eprintln!`. Never `unwrap()`, never `unwrap_or_default()`.

### 2. JSON round-trip destroys NaN — store the real observation

**Symptom:** `TypeError: xy not list of [f64, f64]: must be real number, not NoneType`

**Root cause:** `f64::NAN` → `serde_json::json!(null)` → Python `json.loads()` → `None`. When the bridge extracted `Vec<Vec<f64>>` from a list containing `None`, PyO3 crashed.

**Fix:** Store the concrete `BrightestPointObservation` in the pyclass. `process_image` stashes it. `annotate_image` uses it directly. The JSON dict returned to Python is for the caller's use only — never used for drawing.

**Rule:** Annotation must use the real Rust observation. JSON is lossy. Never reconstruct drawing data from JSON.

### 3. `Box<dyn Observation>` is not `Sync` — PyO3 rejects it

**Symptom:** Compile error: `(dyn Observation + 'static) cannot be shared between threads safely`

**Root cause:** PyO3 requires all pyclass fields to be `Sync`. `Box<dyn Observation>` doesn't implement `Sync` because the `Observation` trait doesn't require it.

**Fix:** Store the concrete type directly:
```rust
struct PyBrightestPointTracker {
    inner: BrightestPointTracker,
    last_obs: Option<BrightestPointObservation>,  // concrete, not Box<dyn>
}
```

**Rule:** Pyclass fields must be concrete types. No `Box<dyn Trait>` in pyclass structs unless the trait has `Send + Sync` bounds.

### 4. Duplicate annotation code — two implementations WILL diverge

**Symptom:** Blob outlines showed in Rust-native tests but not from Python. The PyO3 bridge had its own drawing loop with hardcoded marker parameters.

**Fix:** Extract `draw_markers_into(&mut Mat, &dyn Observation)` as a public method on the tracker. Both the trait impl and the PyO3 bridge call it. One source of truth.

**Rule:** Drawing logic lives in ONE method. The bridge converts data types (numpy↔Mat) but delegates drawing to the tracker.

### 5. Frame copies add up — measure allocations per frame

**Symptom:** `OutOfMemoryError` / `ArrayMemoryError` after running for a few minutes.

**Root causes found:**
- `BaseTracker.process_image()` records every observation into `self.recorder.observations` (unbounded list) by default. `record_observation=False` must be passed in demo loops.
- `Mat::clone()` + `data.to_vec()` = two 2.7MB allocations per frame just for annotation
- `cv2.imshow()` on Windows can hold frame buffer references that Python GC doesn't reclaim fast enough

**Fixes:**
- Pass `record_observation=False` in demo hot loops
- Single `arr.to_owned()` copy for annotation (no `Mat::clone`, no `to_vec`, no `from_shape_fn` closure)
- Periodic `gc.collect()` every 60 frames in the viewer

**Rule:** Every allocation in the per-frame path must be accounted for. One frame copy max for annotation. No unbounded lists. Explicit GC in long-running demos.

### 6. opencv crate `detectBoard` requires pre-populated marker containers

**Symptom:** `detect_board()` / `detect_board_def()` fails with `size_t(i) < vv->size()` assertion in `_InputArray::getMat_()`. Multiple output types tried (`Vector<Point2f>`, `Mat`, `Vector<Mat>`, `Vector<Vector<Point2f>>`) — all fail.

**Root cause:** The C++ `detectBoard()` internally accesses the `InputOutputArray` marker parameters as input (to check format/type) before writing to them. When containers are empty, the C++ assertion fires. The opencv crate binds these as `&mut impl ToInputOutputArray` but cannot pass `noArray()` (the C++ default). Unlike `cv2`'s Python bindings which handle `noArray()` transparently, the Rust opencv crate requires explicit values.

**Fix — decomposed detection:**
1. Run `ArucoDetector::detect_markers_def()` to populate marker vectors first
2. Pass the now-populated marker vectors to `CharucoDetector::detect_board()` as `InputOutputArray`
3. `detectBoard` reads the pre-detected markers and interpolates charuco corners from them

This is what `detectBoard()` does internally anyway (detect markers → interpolate corners). The decomposition makes the steps explicit without changing the algorithm.

**Rule:** When an OpenCV function takes `InputOutputArray` parameters with `noArray()` defaults, pre-populate them before the call. Never pass empty containers to `InputOutputArray` in the opencv crate.

### 7. `ndarray::from_shape_fn` closure overhead — use `to_owned()` for copies

**Symptom:** Rust annotation runs 3ms slower than Python annotation for the same frame. Constant overhead regardless of board visibility.

**Root cause:** `Array3::from_shape_fn(|(y,x,c)| arr[[y,x,c]])` evaluates a Rust closure for every pixel — 2.77 million closure calls per 720p frame. Python's `image.copy()` is a single C-level `memcpy`.

**Fix:** `arr.to_owned()` — does the same data copy via a single `memcpy`. Zero closure overhead.

**Rule:** `from_shape_fn` is for element-wise transforms. For straight copies, always use `to_owned()`. Profile every per-frame allocation.

### 8. Dictionary enum values must be verified across bindings

**Symptom:** Board construction succeeds silently, but detection returns 0 results. No error message — board appears valid but detects nothing.

**Root cause:** Assumed `DICT_4X4_250 = 10` based on the Rust opencv crate constant position. Actually `DICT_4X4_250 = 2` in both Python cv2 and the Rust opencv crate (and OpenCV C++). Passing `10` selects `DICT_6X6_100` — wrong dictionary, wrong markers, zero detections.

**Fix:** Always print the enum value from both Python and Rust and verify they match. Never assume enum values — the Rust opencv crate's `opencv_type_enum!` macro assigns sequential values from 0 matching OpenCV C++, but Python cv2 may remap them.

**Rule:** Verify every config parameter (squares, lengths, dictionary enum) prints the same value in Python and Rust. One mismatched parameter silently breaks detection.

### 9. `Vector<T>::get()` returns `Result<T>`, not `Option<T>`

**Symptom:** Compile error: `mismatched types — expected Result<Point2f, Error>, found Option<_>`.

**Root cause:** The opencv crate's `Vector<T>::get(i)` returns `Result<T, opencv::Error>` — unlike Rust's `Vec::get()` which returns `Option<&T>`. The `get()` method calls into C++ which can fail (out of bounds, type mismatch).

**Fix:** Use `if let Ok(val) = vec.get(i)` instead of `if let Some(val) = vec.get(i)`.

**Rule:** opencv crate container accessors return `Result`, not `Option`. Always use `Ok` / `Err` patterns.

### 10. `Mutex<CharucoTracker>` needed when tracker contains opencv raw-pointer types

**Symptom:** Compile error: `*mut c_void cannot be shared between threads safely — PyCharucoTracker is not Sync`.

**Root cause:** `CharucoBoard` and `CharucoDetector` contain raw C++ pointers (`*mut c_void`) which don't implement `Sync`. BPT avoids this because its fields are primitives (`usize`, `f64`). PyO3 requires all pyclass fields to be `Sync`.

**Fix:** Wrap the tracker in `Mutex<CharucoTracker>`. The GIL ensures single-threaded access in practice; `Mutex` satisfies the `Sync` bound with minimal overhead. Methods lock/unlock the mutex. Same for `last_obs: Mutex<Option<CharucoObservation>>`.

**Rule:** Any tracker whose fields contain opencv types (boards, detectors) must be wrapped in `Mutex<T>` in the PyO3 bridge. Only pure-data trackers (like BPT) can be stored directly.

### 11. beartype checks at runtime — duck-typing is not enough

**Symptom:** `BeartypeCallHintParamViolation: parameter tracker=... not instance of BaseTracker`

**Root cause:** `beartype_this_package()` in `skellytracker/__init__.py` decorates every function in the package. `WebcamDemoViewer.__init__` has `tracker: BaseTracker`. Any object assigned to `self.tracker` MUST pass `isinstance(obj, BaseTracker)`.

**Fix:** `RustBrightestPointTracker(BaseTracker)` — proper subclass with dataclass field stubs.

**Rule:** Every Rust adapter must be a `BaseTracker` subclass. beartype is non-negotiable — it's baked into the package init.

## Patterns to Reuse

### For every new tracker translation:

1. **Create `src/trackers/<name>/mod.rs`** — implement `Tracker` trait
2. **Create `src/trackers/<name>/observation.rs`** — implement `Observation` trait
3. **Add `pub mod <name>` to `src/trackers/mod.rs`**
4. **Add pyclass wrapper in `src/pyo3_bridge/mod.rs`** — or a separate file if large
5. **Add adapter class in Python** — `skellytracker/trackers/<name>/rust_bridge.py`
6. **Add hotkey + backend toggle in `webcam_demo_viewer.py`**

### Checklist per tracker:

- [ ] **Rule #0:** All Python observation fields present in Rust observation (names + types match)
- [ ] `to_json()` outputs every field Python's `to_json_string()` would
- [ ] All OpenCV calls have error handling (no `unwrap_or_default`)
- [ ] Drawing constants extracted at module level
- [ ] `draw_markers_into()` is the single drawing entry point
- [ ] PyO3 bridge stores concrete observation (not `Box<dyn>`)
- [ ] `annotate_image` uses stored obs, not JSON-reconstructed data
- [ ] Annotation does one `to_owned()` copy max (no `from_shape_fn` closure, no `Mat::clone`)
- [ ] Annotation reuses buffers across markers (Vector, String) — no per-element allocs
- [ ] Adapter subclasses `BaseTracker` (beartype compatible)
- [ ] `.create()` classmethod on adapter
- [ ] `record_observation=False` in demo viewer
- [ ] Hotkey toggles backend instantly, respects current tracker type
- [ ] NOT IMPLEMENTED warning for trackers without Rust backend
- [ ] Dictionary enum values verified identical in Python and Rust
- [ ] Board geometry stored in observation (object coordinates for calibration)
- [ ] `Mutex<T>` wrapping if tracker contains opencv types
- [ ] `cargo check` compiles before `poe rebuild`

## Constraints

1. **OpenCV 4.13.0** — installed at `C:/tools/opencv/`, configured via `.cargo/config.toml`
2. **opencv crate 0.98** — API differs from Python cv2 in parameter order and return types
3. **ndarray 0.16** — replaces numpy, same row-major layout, different slice syntax (`s![.., ..2]`)
4. **PyO3 0.23** — `get_item` returns `PyResult<Option<Bound>>` (double-wrapped)
5. **beartype** — always on, always type-checking. No escape hatches in this codebase.
6. **maturin 1.x** — `module-name` must match `#[pymodule]` name exactly
7. **Edition 2021** — not 2024 like skellycam (toolchain compatibility)
8. **No `log` crate** — use `eprintln!` for Rust-side logging (avoids adding deps)

## Timeline

| Phase | Duration | What |
|-------|----------|------|
| Core traits + PointCloud + Recorder | ~2h | Foundation |
| BrightestPointTracker | ~1h | Detection + annotation |
| PyO3 bridge (initial) | ~1h | Working, but with bugs |
| PyO3 bridge (fixes) | ~2h | NaN→None crash, `last_obs` storage, Sync constraint |
| Hot-swappable adapter | ~1h | `RustBrightestPointTracker`, WebcamDemoViewer integration |
| Audit + hardening | ~2h | Error handling, contour outlines, memory fixes, docs |
| **BrightestPoint Total** | **~9h** | Complete, production-quality |
| | | |
| CharucoTracker detection + observation | ~2h | Board construction, detectMarkers→detectBoard, 18-field data model |
| CharucoTracker annotation | ~1h | Aruco boxes, corner markers, text labels, undetected list |
| PyO3 bridge + Mutex wrapping | ~1h | Sync constraint from CharucoBoard raw pointer |
| detectBoard output type debugging | ~3h | Vector<Point2f>→Mat→Vector<Mat>→Vector<Vector<T>> — settled on pre-populated markers |
| Data model parity audit | ~1h | Added 9 missing fields, object coordinates, deferred pose fields |
| Annotation perf optimization | ~1h | Reusable buffers, to_owned() vs from_shape_fn, String reuse |
| Hot-swappable adapter + demo | ~0.5h | RustCharucoTracker, r-key within-Charuco toggle |
| **Charuco Total** | **~9.5h** | Complete, production-quality, 100% detection parity |
| | | |
| **Grand Total** | **~18.5h** | Two trackers translated |

Expected for RTMPose: ~6-8h (ONNX Runtime complexity, batched inference, different output shapes).
