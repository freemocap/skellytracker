# CompositeGPU Handoff — Next Tracker Translation

> For the next agent picking up CompositeGPU translation after BrightestPoint, Charuco, RTMPose (Phase 2 GPU), and MediaPipe (Phase 1) are complete.

## Where we are

Four trackers translated, patterns proven across 3 bridge strategies:

| Tracker | Detection backend | Complexity | Bridge pattern |
|---------|------------------|------------|----------------|
| BrightestPoint ✅ | OpenCV `findContours` | Simple | Pure Rust + pyclass |
| Charuco ✅ | OpenCV `detectBoard` | Medium | Mutex-wrapped (raw C++ ptrs) |
| RTMPose ✅ Phase 2 | ONNX Runtime CUDA (YOLOX + RTMPose) | Hard | `ort` crate, CUDA EP, ~25ms/frame |
| MediaPipe ✅ Phase 1 | Python `mediapipe` via PyO3 reverse bridge | Medium | `Py<PyAny>` refs, `call_method1` |
| **CompositeGPU** 🔜 | **ONNX Runtime CUDA (RTMO body + hand + face)** | **Very Hard** | **Multi-model session pool, batched inference, ROI pipeline** |

## What makes this different from RTMPose

The existing Rust RTMPose tracker is a **single two-stage pipeline** (YOLOX detection → RTMPose wholebody keypoints, 133 output points). CompositeGPU is a **three-model composable pipeline** (RTMO body → ROI crops → parallel hand + face inference, 165 output points) with model selection via presets.

| Characteristic | Existing RTMPose Rust | CompositeGPU target |
|---------------|----------------------|---------------------|
| **Number of ORT sessions** | 2 (det + pose) | 3 (body + hand + face) |
| **Models** | Fixed pair | Swappable via presets (light/medium/heavy) |
| **Inference pattern** | Per-image sequential | Batched (N images → 1 session.run call per model) |
| **ROI cropping** | None (affine crop from YOLOX bbox) | Square ROI from body landmarks (EMA-smoothed, forearm-projected) |
| **Hand processing** | None | Anthropometry validation, wrist blending, overlap dedup, adaptive size memory |
| **Output points** | 133 (COCO-WholeBody) | 165 (RTMO body 17 + right hand 21 + left hand 21 + face 106) |
| **Preprocessing modes** | 1 (RTMPose letterbox) | 2 (RTMO letterbox for body, MediaPipe RGB for hands, RTMPose letterbox for face) |
| **Output decoding** | SIMCC argmax | SIMCC argmax (face) + RTMO NMS (body) + direct regression (MediaPipe hands) |
| **Model config** | Inline `mode_config()` | `ModelSpec` frozen structs with `ModelSource` URLs, preprocessing dispatch |
| **Thread parallelism** | None | `rayon` for parallel hand+face inference |

## What CompositeGPU does (Python architecture)

```
BGR Images [N] → [RTMO letterbox preprocess × N] → stack (N,3,640,640) → [RTMO ONNX]
    │                                                                           ↓
    │                                                            RTMO postprocess (NMS + keypoints)
    │                                                                           ↓
    │                                                          body_kpts: (N, 17, 2)  +  scores
    │                                                                           │
    ├─ Wrist positions → compute hand ROI squares (smoothed, forearm-projected)
    │    │
    │    ├─ Left hand crops → [Hand ONNX] → SIMCC decode or direct regression → (21, 2)
    │    ├─ Right hand crops → [Hand ONNX] → same
    │    ├─ Hand post-processing: anthropometry filter, wrist blend, overlap dedup, adaptive size
    │    └─ Hands run IN PARALLEL (ThreadPoolExecutor with GIL-released MediaPipe C++)
    │
    └─ Head landmarks → compute face ROI square (smoothed, downward-shifted)
         │
         └─ Face crops → [Face ONNX] → SIMCC decode → (106, 2)
                                                                                   ↓
                                          [MERGE] → 165-point PointCloud (body + r_hand + l_hand + face)
```

### Model details

| Component | Model (medium preset) | Input | Output | Preprocess |
|-----------|----------------------|-------|--------|------------|
| Body | RTMO-M | (640, 640) BGR | 17 COCO keypoints + bboxes | RTMO letterbox + BGR mean/std |
| Hand | MediaPipe Hand Landmark | (224, 224) RGB | 21 keypoints × 2 hands | BGR→RGB, resize, /255.0 |
| Face | RTMPose Face LaPa 106 | (256, 256) BGR | 106 keypoints | Simple letterbox + BGR mean/std |

**Total composed**: 165 points (17 body + 21 right_hand + 21 left_hand + 106 face)

### Preset system

`CompositeGPUSessionConfig.preset(tier)` swaps only the body model:

| Tier | Body model | Hand model | Face model |
|------|-----------|------------|------------|
| `light` | RTMO-S | MediaPipe Hand | RTMPose Face 106 |
| `medium` (default) | RTMO-M | MediaPipe Hand | RTMPose Face 106 |
| `heavy` | RTMO-L | MediaPipe Hand | RTMPose Face 106 |

Hand and face models can be individually overridden via `body_spec`/`hand_spec`/`face_spec` fields.

### Key pipeline details

1. **Body first** (serial dependency): Must know wrist/head positions before cropping hands/face.
2. **Hands + face in parallel**: Python uses `ThreadPoolExecutor(max_workers=2)`. MediaPipe C++ hand model releases the GIL; RTMPose face model doesn't. Parallelism still helps — preprocessing + inference of hand crops overlaps with face inference.
3. **ROI computation** (`roi_crop_utils.py`): Pure geometry — forearm projection vectors, EMA smoothing, square crop clamping, downward face shift. No I/O, no ML — just math.
4. **Hand post-processing** (`_hand_postprocess()`): Configurable validation pipeline — anthropometry bounds check, wrist blending (confidence-weighted average of hand root + body wrist), overlap dedup (when left/right detections land on the same hand), adaptive size memory (EMA of valid hand sizes, rejects outliers).
5. **Two preprocessing modes**: MediaPipe hand model uses BGR→RGB + resize → [0,1] scaling. RTMPose face uses simple letterbox + BGR mean/std normalization. The output decoding strategy depends on `preprocess_mode`.
6. **Two hand model backends**: The default is MediaPipe Hand Landmark (direct regression, ONNX exported from PINTO model zoo). The alternative is RTMPose-M Hand5 (SIMCC-based, same architecture as face).
7. **Batched inference**: When `supports_batching`, N images are preprocessed, stacked into one tensor, and sent through a single `session.run()` call. This amortizes ORT launch overhead and maximizes GPU utilization.

### Model download (`model_registry.py`)

Models come from two sources:
- **OpenMMLab CDN** (most models): .zip files with .onnx inside. Download + extract + cache to `~/.cache/skellytracker/models/`.
- **Hugging Face Hub** (MediaPipe models): `hf_hub_download()` via `huggingface_hub` package.

Downloads run in parallel via `ThreadPoolExecutor`. Cached models skip re-download.

### Key files

```
skellytracker/trackers/composite_gpu_tracker/
├── __composite_gpu_tracker.py                  # CompositeGPUTracker (BaseTracker) + create()
├── composite_gpu_detector.py                   # Thin detector wrapper → session
├── composite_gpu_session.py       (1168 lines)  # THE CORE: 3 sessions, batched inference, ROI, hand validation
├── composite_gpu_observation.py                # 165-point PointCloud assembly
├── composite_gpu_annotator.py     (183 lines)   # Skeleton drawing with per-component colors
├── composite_gpu_config.py                     # Pydantic configs + preset factory
├── roi_crop_utils.py              (185 lines)   # Pure geometry: ROIBox, smoothing, crop extraction
├── sub_model_spec.py                           # Backward-compat re-exports
├── names_and_connections/
│   ├── rtmo_body_17.yaml                       # 17 COCO body keypoints + 16 connections
│   ├── rtmpose_face_106.yaml                   # 106 LaPa face keypoints (unprefixed)
│   ├── rtmo_hybrid.yaml                        # Composition: body(17) + r_hand(21) + l_hand(21) + face(106) = 165
│   └── mediapipe_hand.yaml                     # 21 hand keypoints (used with "right_hand_" / "left_hand_" prefix)

skellytracker/utilities/gpu_utils/
├── model_registry.py              (578 lines)   # ModelSource, ModelSpec, TrackerPreset, URL registry, parallel download
├── ort_session_utils.py           (415 lines)   # ORT session builder, CUDA/TRT options, batch probes, dynamic-batch ONNX
├── rtm_preprocessing.py           (301 lines)   # RTMO/RTMPose/YOLOX letterbox, affine warp, normalize
└── rtm_postprocessing.py          (164 lines)   # SIMCC decode, NMS, multiclass_nms
```

### The session file in detail (`composite_gpu_session.py`)

This is 70% of the translation complexity. At 1168 lines it handles:

1. **`.create()` classmethod**: Parallel model download → resolve anatomical indices → build 3 ORT sessions → probe batch support → warmup
2. **`predict_batch()`**: Main entry point. Runs body batch → parallel hands+face → merge
3. **`_run_body_batch()`**: RTMO preprocess → stack → session.run() → RTMO postprocess per image
4. **`_run_hands_batch()`**: For each valid body → compute left+right ROIs → preprocess crops per mode → stack or sequential → session.run() → decode per mode → apply ROI offset → hand post-processing pipeline
5. **`_run_face_batch()`**: For each visible head → compute face ROI (downward shift, EMA smooth, size clamp) → preprocess → stack or sequential → session.run() → SIMCC decode → apply ROI offset
6. **`_hand_postprocess()`**: Per-hand validation pipeline (anthropometry, wrist blend, overlap dedup, adaptive size)
7. **Cache helpers**: `_cache_session_io_names()`, `_prep_scratch_tensor_cache()`

### The observation data model

```python
@dataclass(slots=True)
class CompositeGPUObservation(BaseObservation):
    tracker_type: str = "rtmo_hybrid"
    frame_number: int
    image_size: tuple[int, int]
    points: PointCloud                                # (165, 3) with (165,) visibility
    body_keypoints: NDArray                           # (num_persons, 17, 2) float64
    body_scores: NDArray                              # (num_persons, 17) float32
    hands_keypoints: NDArray                          # (1, 42, 2) — right(0:21) then left(21:42)
    hands_scores: NDArray                             # (1, 42)
    face_keypoints: NDArray                           # (1, 106, 2)
    face_scores: NDArray                              # (1, 106)
    raw_hands_keypoints: NDArray                      # (1, 42, 2) pre-cleanup
    raw_hands_scores: NDArray                         # (1, 42)
    right_hand_roi: ROIBox | None
    left_hand_roi: ROIBox | None
    face_roi: ROIBox | None
```

## Architecture for the Rust side

### Files to create

```
skellytracker-rust/src/
├── onnx_utils/
│   ├── mod.rs                     # Extend: ModelSpec struct, TrackerPreset enum, registry, parallel downloads
│   ├── model_registry.rs          # NEW: ModelSource, ModelSpec, TrackerPreset, URL list, resolve_models_parallel()
│   ├── session_builder.rs         # Extend: add ensure_dynamic_batch_onnx port (or keep as-is — CUDA doesn't need it)
│   ├── preprocessing.rs           # Extend: add rtmo_preprocess(), rtmo_postprocess(), decode_mediapipe_hand()
│   └── postprocessing.rs          # Extend: add rtmo_nms_postprocess() (RTMO uses different NMS from YOLOX)
├── trackers/
│   └── composite_gpu/
│       ├── mod.rs                 # CompositeGpuTracker struct + Tracker impl + detect() pipeline
│       ├── observation.rs         # CompositeGpuObservation (165 points, body/hands/face sub-arrays)
│       ├── session.rs             # CompositeGpuSession — 3 ORT sessions, batched inference, ROI, hand validation
│       ├── roi.rs                 # ROIBox, square ROI computation, EMA smoothing, crop extraction (from roi_crop_utils.py)
│       └── hand_postprocess.rs    # Anthropometry filter, wrist blending, overlap dedup, adaptive size memory
└── pyo3_bridge/
    └── mod.rs                     # + PyCompositeGpuTracker pyclass
```

### Files to modify

```
skellytracker-rust/src/trackers/mod.rs                          # + pub mod composite_gpu;
skellytracker/trackers/composite_gpu_tracker/
└── rust_bridge.py              # RustCompositeGpuTracker(BaseTracker) adapter + factory
skellytracker/io/demo_viewers/webcam_demo_viewer.py              # + g-key + p-key toggle
```

### Architecture diagram

```
Rust CompositeGpuTracker
  │
  ├─ session: CompositeGpuSession
  │    ├─ body_session: ort::Session     (RTMO, 640×640, CUDA EP)
  │    ├─ hand_session: ort::Session     (MediaPipe Hand, 224×224, CUDA EP)
  │    ├─ face_session: ort::Session     (RTMPose Face, 256×256, CUDA EP)
  │    │
  │    ├─ body_spec: ModelSpec           (preprocess mode, mean/std, input size)
  │    ├─ hand_spec: ModelSpec
  │    ├─ face_spec: ModelSpec
  │    │
  │    ├─ Hand ROI state (EMA-smoothed, frame-persistent):
  │    │   smooth_left_center, smooth_right_center, smooth_face_roi
  │    │   smooth_left_hand_diag, smooth_right_hand_diag
  │    │
  │    └─ Thread pool: rayon::ThreadPool (for parallel hand+face inference)
  │
  ├─ detect(frame_number, &Mat) → CompositeGpuObservation:
  │    1. Body: rtmo_preprocess → body_session.run() → rtmo_postprocess → 17 keypoints
  │    2. Compute hand ROIs from wrist positions (roi.rs)
  │    3. Hands + Face IN PARALLEL (rayon scope):
  │       a. Extract + preprocess hand crops → hand_session.run() → decode → ROI offset
  │       b. Extract + preprocess face crop  → face_session.run() → SIMCC decode → ROI offset
  │    4. Hand post-processing (hand_postprocess.rs)
  │    5. Build 165-point PointCloud
  │
  └─ draw_markers_into(&mut Mat, &dyn Observation):
       Per-component skeleton drawing (body=green, r_hand=red, l_hand=blue, face=cyan)
       ROI box overlays + raw-hands ghost drawing
```

### The observation data model (Rust)

```rust
pub struct CompositeGpuObservation {
    pub tracker_type: &'static str,      // "rtmo_hybrid"
    pub frame_number: u64,
    pub image_size: (u32, u32),
    pub points: PointCloud,              // 165 points in hybrid YAML order
    // Sub-component arrays:
    pub body_keypoints: Array3<f64>,     // (1, 17, 2)
    pub body_scores: Array2<f32>,        // (1, 17)
    pub hands_keypoints: Array3<f64>,    // (1, 42, 2)  right[0:21], left[21:42]
    pub hands_scores: Array2<f32>,       // (1, 42)
    pub face_keypoints: Array3<f64>,     // (1, 106, 2)
    pub face_scores: Array2<f32>,        // (1, 106)
    pub raw_hands_keypoints: Array3<f64>, // (1, 42, 2) pre-cleanup
    pub raw_hands_scores: Array2<f32>,   // (1, 42)
    // ROI boxes (None when not computed):
    pub right_hand_roi: Option<RoiBox>,
    pub left_hand_roi: Option<RoiBox>,
    pub face_roi: Option<RoiBox>,
}
```

## What can be reused from existing Rust code

| Component | File | Status |
|-----------|------|--------|
| `PointCloud` struct | `point_cloud.rs` | ✅ Fully reusable |
| `Observation` / `Tracker` traits | `traits.rs` | ✅ No changes needed |
| `Provider` enum + CUDA/TRT EP builders | `session_builder.rs` | ✅ TRT EP for body — same YOLOX-NMS-hang issue; use CUDA for body, CUDA for hand/face |
| YOLOX letterbox preprocess | `preprocessing.rs` | ✅ Reusable (same algo, different input size) |
| SIMCC `get_simcc_maximum` | `postprocessing.rs` | ✅ Reusable for face model |
| `nms()` | `postprocessing.rs` | ✅ Reusable for body model |
| Top-down affine warp | `preprocessing.rs` | ✅ Reusable for face crops |
| `resolve_model()` | `onnx_utils/mod.rs` | ⚠️ Needs generalization for parallel downloads + HF Hub |
| `mat_to_numpy` / numpy→Mat | `mediapipe/mod.rs` | ⚠️ Only needed if annotation delegates to Python (unlikely — implement in Rust) |
| PyO3 bridge pattern | `pyo3_bridge/mod.rs` | ✅ Same `Mutex<T>` pattern (ort::Session needs `&mut self`) |
| Python adapter pattern | `rtmpose_tracker/rust_bridge.py` | ✅ Copy + adapt for CompositeGPU |

## Implementation phases

### Phase 1: Infrastructure (model registry + preprocessing)

- Port `model_registry.py` → `model_registry.rs`: `ModelSource`, `ModelSpec` (frozen config struct with `preprocess_mode` enum), `TrackerPreset` enum, URL registry, `resolve_models_parallel()` with `rayon` for concurrent downloads
- Port `rtmo_preprocess()` and `rtmo_postprocess()`: RTMO letterbox (640×640, BGR mean/std), RTMO output decoder (NMS + bbox → 17×2 keypoints + scores)
- Port `decode_mediapipe_hand()`: extract `xyz_x21` and `hand_score` named tensors from MediaPipe hand ONNX output, scale from model coords to crop-pixel coords
- Port `roi_crop_utils.py` → `roi.rs`: `RoiBox` struct, `compute_square_roi()`, EMA smoothing helpers, forearm projection, downward face shift
- Add `ensure_dynamic_batch_onnx` for the RTMO body model (or verify CUDA doesn't need it)

### Phase 2: Core session + pipeline

- Port `composite_gpu_session.py` → `session.rs`: `CompositeGpuSession` struct with 3 `ort::Session` fields, `create()` classmethod, `predict_batch()`, `_run_body_batch()`, `_run_hands_batch()`, `_run_face_batch()`
- Cache I/O names at session build time (like Python does)
- Rayon scope for parallel hand+face inference: `py.allow_threads(|| pool.scope(|s| { ... }))`
- Hand post-processing pipeline → `hand_postprocess.rs`: `_hand_postprocess()` with configurable validators (anthropometry bounds, wrist blending, overlap dedup, adaptive size memory)

### Phase 3: Observation + annotation + bridge

- Port `composite_gpu_observation.py` → `observation.rs`: `CompositeGpuObservation` with Rule #0 data model parity, 165-point PointCloud assembly
- Port `composite_gpu_annotator.py` → `mod.rs` `draw_markers_into()`: per-component skeleton (body/hands/face with different colors), ROI boxes, raw-hands ghost overlay
- Add `PyCompositeGpuTracker` pyclass in `pyo3_bridge/mod.rs` (Mutex-wrapped like RTMPose)
- Add `RustCompositeGpuTracker(BaseTracker)` adapter in `composite_gpu_tracker/rust_bridge.py`
- Add `g` hotkey + p-key toggle in webcam demo viewer

### Phase 4 (Optional): Advanced features

- TRT execution provider for face model (simpler than YOLOX RTMPose — clean feed-forward, may not hang)
- Configurable hand/face model swapping via `ModelSpec` at runtime
- `TrackerPreset` shorthand (`light`/`medium`/`heavy`)
- Multi-person body support (currently first person only)

## Critical rules from lessons learned

All 23 rules from the existing lessons learned apply. Especially relevant for CompositeGPU:

- **Rule #0** — Data model parity: Rust observation MUST have every field the Python observation has
- **Rule #6** — OpenCV `unwrap_or_default()` is a trap: error-check every OpenCV call
- **Rule #14** — `ort::Session::run()` takes `&mut self`: tracker needs `Mutex<T>` wrapping in pyclass
- **Rule #16** — Affine warp: always use `getAffineTransform`, never hand-roll the matrix
- **Rule #20** — Mat↔numpy round-trips cost one copy per direction: implement annotation in Rust (skip Python annotator delegation)
- **Rule #22** — Test GPU EP on Python side first: cross-validate CUDA behavior before debugging Rust
- **Rule #23** — Models with baked NMS hang TRT: RTMO body has NMS baked in (like YOLOX) — use CUDA for body

New considerations for this tracker:

24. **ORT session pools need coordinated lifecycle** — Three `ort::Session` objects share one CUDA context. All must be built with the same `gpu_mem_limit` so the arena allocator distributes memory across them. Build order doesn't matter, but all three must exist before any inference runs.

25. **Batched inference has different tensor shapes** — When N>1, input tensors are (N, C, H, W) not (1, C, H, W). Output processing must iterate over the batch dimension. When `supports_batching` is false, fall back to sequential per-crop `session.run()` with manual stacking.

26. **Hand ROI cropping depends on body keypoint indices** — These come from the YAML definition, not hardcoded constants. Use `RTMO_BODY_17_DEFINITION.index_of("left_wrist")` to resolve indices dynamically so body model swaps work.

27. **Preprocessing mode dispatch determines output decoding** — The `ModelSpec.preprocess_mode` field controls BOTH input normalization AND output decoding strategy. Don't split these — keep them as a single dispatch.

## Reference implementation to study

1. **Python composite GPU session** — `skellytracker/trackers/composite_gpu_tracker/composite_gpu_session.py` (1168 lines — the core)
2. **Python model registry** — `skellytracker/utilities/gpu_utils/model_registry.py` (578 lines — ModelSpec, URLs, downloads)
3. **Python ORT session builder** — `skellytracker/utilities/gpu_utils/ort_session_utils.py` (415 lines — already ~60% ported in `session_builder.rs`)
4. **Python ROI crop utilities** — `skellytracker/trackers/composite_gpu_tracker/roi_crop_utils.py` (185 lines — pure geometry)
5. **Python composite observation** — `skellytracker/trackers/composite_gpu_tracker/composite_gpu_observation.py` (130 lines — 165-point assembly)
6. **Python composite annotator** — `skellytracker/trackers/composite_gpu_tracker/composite_gpu_annotator.py` (183 lines — skeleton drawing)
7. **Python composite config** — `skellytracker/trackers/composite_gpu_tracker/composite_gpu_config.py` (44 lines — config hierarchy)
8. **Rust RTMPose tracker** — `skellytracker-rust/src/trackers/rtmpose/mod.rs` (Rust Tracker pattern to follow)
9. **Rust RTMPose observation** — `skellytracker-rust/src/trackers/rtmpose/observation.rs` (data model pattern)
10. **Rust session builder** — `skellytracker-rust/src/onnx_utils/session_builder.rs` (CUDA/TRT EP builders — reusable)
11. **Rust preprocessing** — `skellytracker-rust/src/onnx_utils/preprocessing.rs` (letterbox, affine warp — reusable)
12. **Rust postprocessing** — `skellytracker-rust/src/onnx_utils/postprocessing.rs` (SIMCC, NMS — reusable)
13. **Rust PyO3 bridge** — `skellytracker-rust/src/pyo3_bridge/mod.rs` (bridge pattern — reusable)
14. **Python RTMPose adapter** — `skellytracker/trackers/rtmpose_tracker/rust_bridge.py` (adapter pattern — reusable)
15. **Re-architecture docs** — `rearchitecture-docs/skellytracker-architecture/05-lessons-learned.md` (all 23 lessons)
16. **CompositeGPU README** — `skellytracker/trackers/composite_gpu_tracker/composite_gpu_README.md` (388 lines — detailed architecture doc)

## Verification

1. `python -m skellytracker` — press `g` key, verify body/hands/face all draw correctly
2. `p`-key toggle: Python↔Rust adapter produce identical skeleton output
3. Frame time: Rust adapter should match or beat Python GPU performance (~20-35ms/frame depending on model tier)
4. All 165 keypoints present in JSON output, point names match `rtmo_hybrid.yaml`
5. Preset switching (`light`/`medium`/`heavy`) loads different body models
6. Hand post-processing filters bogus detections (test by occluding hands in webcam)
7. Batch inference works (no errors, correct shapes) when `supports_batching=true`
