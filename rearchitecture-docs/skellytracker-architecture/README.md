# SkellyTracker Rust — Architecture Documentation

> Worked example of applying the [re-architecture playbook](../rearchitecture-playbook/) to the skellytracker pose-estimation backend, starting with the BrightestPointTracker as a warm-up.

## Status

| Tracker | Status | Notes |
|---------|--------|-------|
| **BrightestPoint** | ✅ Complete | Full Python→Rust translation with hot-swappable backends |
| Charuco | 🔜 Next | OpenCV-based, follows same pattern |
| RTMPose | 🔜 Planned | ONNX Runtime — most complex, will stress-test the trait design |
| MediaPipe Holistic | 🔜 Later | MediaPipe bindings |
| CompositeGPU | 🔜 Later | Multi-model pipeline |

## Documents

| # | Document | What it covers |
|---|----------|---------------|
| 01 | [Tracker Trait Architecture](./01-tracker-trait-architecture.md) | `Tracker` / `Detector` / `Annotator` / `Observation` traits, `PointCloud`, `Recorder` — the core framework |
| 02 | [BrightestPointTracker Translation](./02-brightest-point-translation.md) | Python → Rust side-by-side: detection, annotation, contour drawing, error handling |
| 03 | [PyO3 Bridge Pattern](./03-pyo3-bridge-pattern.md) | `_skellytracker_rust` native module, `pyo3_bridge/` layout, numpy↔Mat interop, contour data flow |
| 04 | [Hot-Swappable Backend](./04-hot-swappable-backend.md) | `USE_RUST_BACKEND` flag, `RustBrightestPointTracker` adapter, `BaseTracker` subclassing for beartype |
| 05 | [Lessons Learned](./05-lessons-learned.md) | Mistakes, gotchas, patterns to reuse — what to do and NOT do for the next tracker |

## Key Constraints Discovered

1. **beartype runtime type checking is active across the entire package** — any Rust adapter MUST subclass `BaseTracker` or be accepted by the type system at the boundary. Duck-typing alone fails at runtime.

2. **`f64::NAN` → JSON `null` → Python `None`** — the JSON serialization round-trip through the PyO3 bridge destroys NaN values. The observation MUST be stored in Rust and passed directly to the annotator, bypassing the JSON path for drawing.

3. **OpenCV `unwrap_or_default()` is a trap** — it silently swallows errors. Use explicit `is_err()` checks with `eprintln!` logging. Never ignore OpenCV failures in a detection hot loop.

4. **Frame copies add up** — every `Mat::clone()` or `data.to_vec()` is a 2.7MB allocation at 720p. The annotation path must do exactly one copy: source numpy → writable buffer → draw → return.

## Build & Test

```bash
poe rebuild                                    # Rebuild Rust + Python
python skellytracker-rust/webcam_demo.py       # Webcam demo (Rust default)
python skellytracker-rust/webcam_demo.py --python  # Python fallback

# Hotkeys in demo:
#   b — switch to BrightestPointTracker
#   r — toggle Rust ↔ Python (instant swap for BrightestPoint)
#   m — switch to MediaPipe
#   c — switch to Charuco
#   h — show controls    i — toggle info    q — quit
```
