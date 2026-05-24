# Lessons Learned

> Mistakes, gotchas, and patterns to reuse. Everything we'd tell ourselves if we were starting the next tracker translation tomorrow.

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
- Single `ndarray::from_shape_fn` copy for annotation (no `Mat::clone`, no `to_vec`)
- Periodic `gc.collect()` every 60 frames in the viewer

**Rule:** Every allocation in the per-frame path must be accounted for. One frame copy max for annotation. No unbounded lists. Explicit GC in long-running demos.

### 6. beartype checks at runtime — duck-typing is not enough

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

- [ ] All OpenCV calls have error handling (no `unwrap_or_default`)
- [ ] Drawing constants extracted at module level
- [ ] `draw_markers_into()` is the single drawing entry point
- [ ] PyO3 bridge stores concrete observation (not `Box<dyn>`)
- [ ] `annotate_image` uses stored obs, not JSON-reconstructed data
- [ ] Annotation does one frame copy max
- [ ] Adapter subclasses `BaseTracker` (beartype compatible)
- [ ] `.create()` classmethod on adapter
- [ ] `record_observation=False` in demo viewer
- [ ] Hotkey toggles backend instantly
- [ ] NOT IMPLEMENTED warning for trackers without Rust backend
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
| **Total** | **~9h** | Complete, production-quality BrightestPointTracker translation |

Expected for subsequent trackers: ~3-5h each (Charuco simpler, RTMPose more complex due to ONNX).
