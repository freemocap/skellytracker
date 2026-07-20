# Composite GPU Tracker

Composable GPU-accelerated pose tracker that runs body, hand, and face
detection as **separate ONNX models** under a single ONNX Runtime CUDA
context, with batched inference for multi-camera pipelines.

```
┌──────────┐   b  ┌──────────────────────────┐
│  Image   │────▶ │  RTMO Body (640×640)     │
│ (B,G,R)  │      │  17 COCO keypoints       │
└──────────┘      └──────┬────────┬──────────┘
                       │          │
                  wrist xy      head landmarks
                       │          │
              ┌────────▼──┐    ┌──▼────────────┐
              │ Hand Crop  │   │ Face Crop      │
              │ (square ROI)│  │ (square ROI)   │
              └──────┬─────┘   └──┬─────────────┘
                     │              │
          ┌──────────▼──────┐  ┌──▼────────────────┐
          │ Hand Landmark    │  │ Face Landmark      │
          │ 21 kpt × 2 hands │  │ 106 kpt            │
          │ 224×224 (RGB)    │  │ 256×256 (BGR)      │
          └────────┬─────────┘  └──┬─────────────────┘
                   │               │
          ┌────────▼───────────────▼────┐
          │  Merge → 165-point PointCloud │
          │  (body 17 + r_hand 21 +       │
          │   l_hand 21 + face 106)       │
          └──────────────────────────────┘
```

## Quickstart

```python
from skellytracker.trackers.composite_gpu_tracker import CompositeGPUTracker

# Default config (RTMO body + MediaPipe hands + RTMPose face, CUDA)
tracker = CompositeGPUTracker.create()
observation = tracker.process_image(frame_number=0, image=rgb_image)
annotated = tracker.annotate_image(image=rgb_image, observation=observation)

# Or with a preset tier
from skellytracker.trackers.composite_gpu_tracker.composite_gpu_session import (
    CompositeGPUSessionConfig,
)

light_config = CompositeGPUSessionConfig.preset("light")
tracker = CompositeGPUTracker.create(light_config)

# Demo from CLI
python -m skellytracker composite_gpu
```

## Architecture

### Inference pipeline

1. **Body**: RTMO (one-stage) ONNX model processes the full image at 640×640.
   Produces 17 COCO body keypoints + bounding boxes. No separate person
   detector needed — RTMO is a single-stage model.

2. **Hands & Face** run in parallel via `ThreadPoolExecutor`:

   - **Hand crops**: Square ROIs centered on each wrist, projected outward
     along the forearm direction (wrist − elbow vector). Crop size is derived
     from the smoothed face ROI (prev frame) or falls back to 20% of image
     dimension.
   - **Face crop**: Square ROI centered on visible head landmarks (COCO
     indices 0-4: nose, eyes, ears), shifted 20% downward, with a configurable
     scale factor.

3. Each crop is preprocessed per its model contract and stacked into a batch
   for GPU inference. Outputs are decoded, filtered, and merged into a
   single 165-point `PointCloud`.

### Key components

| File | Role |
|------|------|
| `composite_gpu_session.py` | ONNX session lifecycle, batched inference, ROI cropping |
| `composite_gpu_detector.py` | Wraps the session for the detector interface |
| `composite_gpu_observation.py` | Merges body/hands/face into a `PointCloud` |
| `composite_gpu_annotator.py` | Draws skeleton connections and keypoints |
| `composite_gpu_config.py` | Pydantic config models |
| `sub_model_spec.py` | Backwards-compat re-exports from `model_registry` |
| `roi_crop_utils.py` | ROI box math, smoothing, head-point collection |

## Model Spec System

Each sub-model (body, hand, face) is described by a `ModelSpec` — a Pydantic
model that bundles:

| Field | Purpose |
|-------|---------|
| `source` | Where to get the ONNX file (URL, Hugging Face, local path) |
| `input_size` | Model input tensor `(height, width)` |
| `num_keypoints` | Keypoints per instance |
| `preprocess_mode` | Which preprocessing + decode pipeline to use |
| `mean` / `std` | BGR normalisation constants (RTMPose only) |
| `simcc_split_ratio` | SIMCC label resolution divisor (RTMPose only) |

### `preprocess_mode` determines both input and output handling

| Mode | Preprocessing | Output decode |
|------|--------------|---------------|
| `"rtmo"` | RTMO-specific BGR resize + mean/std | RTMO postprocess (NMS + decode) |
| `"rtmpose_letterbox"` | Letterbox + BGR mean/std | SIMCC argmax + split_ratio scaling |
| `"simple_letterbox"` | Letterbox + optional mean/std | (caller-determined) |
| `"mediapipe"` | BGR→RGB + resize + [0,1] scaling | Direct landmark regression |
| `"none"` | Pass-through | (caller-determined) |

### Available ModelSpec factories

All live on the `ModelSpec` class in `skellytracker.utilities.gpu_utils.model_registry`:

```python
# Body (RTMO one-stage, 17 COCO keypoints)
ModelSpec.rtmo_light()      # RTMO-S, 640×640
ModelSpec.rtmo_medium()     # RTMO-M, 640×640  [default]
ModelSpec.rtmo_heavy()      # RTMO-L, 640×640

# Hand (21 keypoints per hand)
ModelSpec.rtmpose_hand()             # RTMPose-M hand5 SIMCC, 256×256
ModelSpec.mediapipe_hand_landmark()  # MediaPipe Hand Landmark, 224×224  [default]

# Face
ModelSpec.rtmpose_face()              # RTMPose-M LaPa 106-point SIMCC, 256×256  [default]
ModelSpec.mediapipe_face_landmark()   # MediaPipe sparse 6-point, 192×192
ModelSpec.mediapipe_face_detector_short()  # BlazeFace short-range detector

# Body (alternative — single-model wholebody)
ModelSpec.mediapipe_pose_landmark()   # MediaPipe Pose 33-keypoint, 256×256
ModelSpec.mediapipe_palm_detector()   # BlazePalm hand detector, 192×192
```

Factories exist for RTMW wholebody (133-keypoint) and RTMPose body models too —
see `model_registry.py` for the full list.

## Configuration Reference

All config lives on `CompositeGPUSessionConfig` (Pydantic model):

### Execution

| Field | Default | Description |
|-------|---------|-------------|
| `execution_provider` | `"cuda"` | ONNX Runtime EP: `"cuda"`, `"trt"`, `"directml"`, `"cpu"` |
| `engine_cache_dir` | `~/.cache/skellytracker/trt_engines` | TensorRT engine cache |
| `max_batch_size` | `4` | Max images per batch |
| `fp16` | `True` | Use FP16 precision (CUDA / TRT only) |
| `on_provider_missing` | `"fallback"` | Behaviour when requested EP unavailable |

### Sub-model selection

| Field | Default | Description |
|-------|---------|-------------|
| `body_spec` | `ModelSpec.rtmo_medium()` | Body pose model |
| `hand_spec` | `ModelSpec.mediapipe_hand_landmark()` | Hand landmark model |
| `face_spec` | `ModelSpec.rtmpose_face()` | Face landmark model |
| `detect_hands` | `True` | Enable hand detection |
| `detect_face` | `True` | Enable face detection |

### ROI crop tuning

| Field | Default | Description |
|-------|---------|-------------|
| `hand_roi_face_scale` | `1.5` | Hand crop size = face ROI size × this |
| `hand_roi_image_fraction` | `0.2` | Fallback: hand crop = min(w,h) × this |
| `hand_roi_center_offset` | `0.17` | Forearm projection distance past wrist |
| `hand_wrist_bias` | `1.5` | Hand confidence multiplier in wrist blending |
| `face_roi_scale` | `2.0` | Face crop = head bbox × this |
| `roi_visibility_threshold` | `0.3` | Minimum body keypoint confidence for ROI |
| `roi_smoothing` | `0.7` | EMA alpha for ROI centre smoothing |
| `body_head_indices` | `[0,1,2,3,4]` | COCO indices for face crop derivation |
| `body_left_wrist_index` | `9` | COCO index |
| `body_right_wrist_index` | `10` | COCO index |
| `body_left_elbow_index` | `7` | COCO index |
| `body_right_elbow_index` | `8` | COCO index |

## TrackerPreset Tiers

The `preset()` classmethod bundles body/hand/face model choices:

```python
from skellytracker.trackers.composite_gpu_tracker.composite_gpu_session import (
    CompositeGPUSessionConfig,
)

light  = CompositeGPUSessionConfig.preset("light")   # RTMO-S  + MediaPipe hands + RTMPose face
medium = CompositeGPUSessionConfig.preset("medium")  # RTMO-M  + MediaPipe hands + RTMPose face
heavy  = CompositeGPUSessionConfig.preset("heavy")   # RTMO-L  + MediaPipe hands + RTMPose face
```

**The tier only changes the body model.** Hands and face are the same across
all tiers by default. To customise them further, override the fields after
calling `preset()`:

```python
config = CompositeGPUSessionConfig.preset("heavy")
config.hand_spec = ModelSpec.rtmpose_hand()  # switch back to RTMPose hands
config.face_spec = ModelSpec.mediapipe_face_landmark()  # sparse 6-point face
```

## ROI Cropping Strategy

### Hand ROIs

For each detected body wrist:
1. Compute the forearm vector: **wrist − elbow**
2. Project the crop centre past the wrist: `center = wrist + forearm_dir × (crop_size × 0.17)`
3. Crop size comes from the **smoothed face ROI** × `hand_roi_face_scale` (default 1.5×),
   or falls back to `hand_roi_image_fraction` × min(image_w, image_h)
4. Smooth the centre with EMA (α = 0.7) to reduce jitter
5. Clamp to image bounds

### Face ROI

1. Collect visible head keypoints (COCO indices 0-4: nose, left/right eye, left/right ear)
2. Compute a bounding box of visible head points
3. Centre shifted 20% downward (face extends further below the eyeline than above)
4. Crop size = head width × `face_roi_scale` (clamped to [120, 600])
5. Smooth with EMA (α = 0.7)

## Hand Post-Processing

After inference, each hand detection passes through configurable validation and
cleanup stages:

### Anthropometry filter (`hand_validation_enabled`, default: `True`)

Rejects implausible hand detections before blending. All thresholds are
configurable:

| Field | Default | Description |
|-------|---------|-------------|
| `hand_validation_enabled` | `True` | Master toggle for all anthropometry checks |
| `hand_min_valid_keypoints` | `4` | Minimum non-NaN keypoints required |
| `hand_bbox_diag_min` | `30.0` | Reject hands smaller than this (px diagonal) |
| `hand_bbox_diag_max` | `280.0` | Reject hands larger than this (px diagonal) |
| `hand_aspect_min` | `0.25` | Minimum keypoint bbox aspect ratio (w/h) |
| `hand_aspect_max` | `4.0` | Maximum keypoint bbox aspect ratio (w/h) |
| `hand_finger_palm_ratio_min` | `1.2` | Middle-finger length must be ≥ this × palm width |
| `hand_score_threshold` | `0.15` | Minimum mean keypoint score to keep a detection |

### Wrist blending (`hand_wrist_blend_enabled`, default: `True`)

Confidence-weighted blend between the hand model's root position and the body
model's wrist position. The `hand_wrist_bias` (default 1.5) gives the hand
model extra weight in the blend.

### Overlap dedup (`hand_overlap_dedup_enabled`, default: `True`)

When left and right hand roots are within `hand_overlap_distance` (default 80px),
the hand farther from its expected body wrist is discarded. This prevents
swapped-hand artifacts when hands cross.

### Wrist blending

When both body wrist and hand root are available, the hand root position is
confidence-weighted blended with the body wrist position. The `hand_wrist_bias`
parameter (default 1.5) gives the hand model extra weight, pulling the blended
wrist toward the hand estimate.

## Hand Model Options

The composite tracker supports two hand model backends, selected via `hand_spec`:

| | RTMPose hand | MediaPipe hand |
|---|---|---|
| `ModelSpec` factory | `rtmpose_hand()` | `mediapipe_hand_landmark()` |
| Architecture | CSPNeXt + SIMCC | BlazePalm-derived CNN |
| Input | 256×256 BGR, letterbox + mean/std | 224×224 RGB NCHW, [0,1] |
| Output | SIMCC heatmaps → argmax decode | Direct (x, y, z) regression |
| Keypoints | 21 | 21 (same ordering) |
| Batch support | Yes (native) | Yes (PINTO dynamic-batch ONNX) |
| Source | OpenMMLab CDN | PINTO model zoo (GitHub) |
| Strengths | Fast, GPU-optimised, SIMCC smoothness | Robust to occlusion & fast motion |
| Training data | AIC + COCO + 3 other datasets | Proprietary Google dataset |

**Keypoint ordering is identical** between the two models (wrist → thumb
chain → index → middle → ring → pinky), so no YAML or annotation changes are
needed when switching.

## Batch Infrastructure

All three sub-models are probed at session creation time via `probe_supports_batch()`.
If a model has a static `batch=1` dimension, a **warning is logged** and
inference falls back to per-crop sequential processing.  This ensures
correctness for any model, but with degraded throughput.

To make a static-batch model batchable, use the general-purpose utility:

```python
from skellytracker.utilities.gpu_utils.ort_session_utils import (
    ensure_dynamic_batch_onnx,
)

batchable_path = ensure_dynamic_batch_onnx("path/to/model.onnx")
# Use batchable_path as the ModelSource
```

This rewrites ``dim[0]`` from static 1 to symbolic ``"N"`` on inputs and outputs.
**Reshape surgery is not included** — models with batch-dependent Reshape nodes
(like YOLOX) need additional fixup (see `_yolox_dynamic_batch.py`).

### `supports_batching` field

`ModelSpec` has an optional `supports_batching` field:

- ``None`` (default) — probe at runtime
- ``True`` / ``False`` — explicit declaration

Set it to ``False`` when you know a model is static-batch, to suppress the
inevitable runtime error and go straight to the per-crop fallback.

## Adding a New Model

1. **Add the URL** to `MODEL_URLS` in `model_registry.py`
2. **Add a `ModelSpec` factory** method (e.g. `ModelSpec.my_new_model()`)
3. Set the fields:
   - `source`: URL, HF repo, or local path
   - `input_size`: model input `(H, W)`
   - `num_keypoints`: keypoints per instance
   - `preprocess_mode`: `"mediapipe"` for RGB [0,1] models, `"rtmpose_letterbox"` for SIMCC, etc.
   - `mean` / `std`: if the model uses BGR normalisation
   - `simcc_split_ratio`: if the model uses SIMCC decoding
4. **If adding a new `preprocess_mode`**: add the preprocessing + decode
   branch in `composite_gpu_session.py:_run_hands_batch()` or
   `_run_face_batch()`.

## Model Download & Caching

Models are downloaded on first use and cached in `~/.cache/skellytracker/models/`.
The `resolve_model_path()` function handles:

- **OpenMMLab CDN URLs** (`.zip` containing `.onnx`): downloaded, extracted, cached
- **Direct `.onnx` URLs** (Hugging Face resolve links): downloaded directly
- **Hugging Face Hub** (`hf_repo` + `hf_filename`): via `huggingface_hub` package
- **Local paths**: returned as-is

Set the `SKEL_MODEL_CACHE_DIR` environment variable to override the cache
location.

## Logging

Startup logs (model paths, provider, warmup timing) are at **INFO** level.
Per-frame inference logs are at **DEBUG** level to avoid spamming the console.

A throttled **INFO** summary line appears for the first 3 frames and every
60th frame thereafter:

```
frame    1 | body:  139ms | hands:  175ms | face:  102ms | total:  416ms
frame    2 | body:   16ms | hands:    5ms | face:    7ms | total:   28ms
frame    3 | body:   15ms | hands:    5ms | face:    6ms | total:   26ms
...
frame   60 | body:   17ms | hands:    6ms | face:    8ms | total:   31ms
```

**Note**: the first frame is slow (GPU JIT / cuDNN auto-tune). Steady state
is reached by frame 2–3.

## Timing Benchmarks

Measured on RTX 4060 Mobile, CUDA EP, FP16, batch_size=1 (webcam):

| Stage | First frame | Steady state | Notes |
|-------|------------|--------------|-------|
| Body (RTMO-M, 640×640) | ~140ms | ~16ms | CUDA JIT on first frame |
| Hands (MediaPipe, 224×224, 2 crops) | ~175ms | ~5ms | Batched inference |
| Face (RTMPose-M, 256×256, 1 crop) | ~102ms | ~7ms | First frame includes JIT |
| **Total** | **~416ms** | **~28ms** | ~35 FPS steady state |

With `max_batch_size=4`, warmup runs one batched pass (4 synthetic frames).
Steady-state single-frame latency is unchanged; multi-camera throughput
benefits from batched body inference.

## Debugging

```python
import logging
# Per-frame detail
logging.getLogger("skellytracker.trackers.composite_gpu_tracker").setLevel(logging.DEBUG)
# Per-crop detail (very verbose)
logging.getLogger("skellytracker.trackers.composite_gpu_tracker").setLevel(5)  # TRACE
```
