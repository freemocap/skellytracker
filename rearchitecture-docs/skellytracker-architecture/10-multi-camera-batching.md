# Multi-Camera Batching

This document describes the architecture for running a single `Tracker` across N cameras in one batched GPU call per model, rather than N sequential calls.

---

## Motivation

freemocap recordings always involve multiple synchronized cameras. The baseline approach — call `tracker.process_image()` once per camera per frame — makes N round-trips to the GPU per model per frame. For a 4-camera setup with a top-down YOLOX→RTMPose pipeline, that is 8 GPU calls per frame where 2 would suffice.

Batched inference amortizes GPU launch overhead, allows the model to use SIMD/tensor core parallelism across the batch dimension, and keeps VRAM usage predictable (all cameras share one allocated context rather than N independent inference buffers). The same batching logic benefits CPU users via ONNX Runtime's CPU-side SIMD across the batch.

---

## Design Principles

**The Session owns the GPU; the DetectionStage owns the pipeline.**
The `Session` is the single holder of GPU context and loaded model weights. It exposes a `run_batched()` method that accepts a stacked `(N, 3, H, W)` tensor and returns `(N, ...)` raw outputs. It knows nothing about pipeline structure. All orchestration — which model to call when, how to split results, what to do with crops — lives in `DetectionStage`.

**Temporal state is per-camera and never batched.**
`TrackerState` / `StageState` capture one camera's smoothing history. For N cameras, the caller holds `dict[cam_id, TrackerState]`. The batching only covers the GPU inference step. Preprocessing metadata, bbox smoothing, keypoint filtering, and child-stage dispatch all remain per-camera.

**`detect` stays as a convenience wrapper.**
All existing single-camera call sites continue to work unchanged. `process_image` is a thin wrapper around `process_batch` with a single-element dict. No API is removed.

**Dict keys, not list indices.**
Camera IDs are caller-chosen strings (e.g. `"cam0"`, `"left"`, `"/dev/video2"`). All multi-camera APIs use `dict[str, ...]` rather than `list[...]` to eliminate ordering bugs when stacking and splitting tensors.

---

## The Preprocess / Infer / Postprocess Split

Each detector exposes three methods alongside `detect`:

```python
def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray, Metadata]:
    """Return a float32 tensor ready for the model, plus metadata for postprocess."""

def infer(self, tensor: NDArray, session: Session) -> Any:
    """Run the model on a single-image tensor (1, 3, H, W). Used by detect()."""

def postprocess(self, raw: Any, metadata: Metadata) -> list[BoundingBox] | Keypoints:
    """Decode raw model output back to image-space coordinates."""
```

`DetectionStage.run_batch()` calls `preprocess` and `postprocess` directly and bypasses `infer`, replacing it with a single `session.run_batched()` call across all cameras.

### Metadata

`Metadata` is a small typed dataclass per detector — not `Any` at runtime. It carries only what `postprocess` needs that cannot be recovered from the raw output:

| Detector | Metadata |
|----------|----------|
| YOLOX | `ratio: float`, `original_size: tuple[int, int]` |
| RTMPose | `center: NDArray[float64]`, `scale: NDArray[float64]` |
| MediaPipe | *(empty dataclass)* |
| Charuco / ArUco | *(empty dataclass)* |
| BrightestPoint | *(empty dataclass)* |

Empty metadata is still a typed dataclass (not `None`) so the contract is uniform.

---

## Session: `run_batched()`

```python
class OnnxSession(Session):
    def run_batched(
        self,
        model_name: str,
        tensors: dict[str, NDArray],   # cam_id → (3, H, W) float32
    ) -> dict[str, NDArray]:           # cam_id → raw model output for that camera
        ordered_keys = list(tensors.keys())
        stacked = np.stack([tensors[k] for k in ordered_keys])  # (N, 3, H, W)
        raw = self._ort_sessions[model_name].run(None, {input_name: stacked})[0]  # (N, ...)
        return {k: raw[i] for i, k in enumerate(ordered_keys)}
```

The dict-in / dict-out design means ordered_keys is derived from the input dict, so the stacking and splitting indices always stay in sync. The session never sees bare integer indices.

---

## DetectionStage: `run_batch()`

```python
def run_batch(
    self,
    images: dict[str, NDArray[np.uint8]],
    states: dict[str, StageState],
    context: DetectionContext,
) -> tuple[dict[str, StageObservation], dict[str, StageState]]:
```

### Execution for a top-down stage (ObjectDetector + KeypointDetector)

```
Step 1: Preprocess object detection
    For each cam_id in images:
        tensor[cam_id], obj_meta[cam_id] = object_detector.preprocess(images[cam_id])
    → Optionally vectorized if all cameras share a resolution

Step 2: Batched object detection  [one GPU call]
    raw_obj = session.run_batched(object_detector.model_name, tensor)

Step 3: Postprocess + bbox smoothing per camera
    For each cam_id:
        bboxes[cam_id] = object_detector.postprocess(raw_obj[cam_id], obj_meta[cam_id])
        smooth_bbox[cam_id], states[cam_id] = apply_bbox_ema(bboxes[cam_id], states[cam_id])

Step 4: Compute crops per camera
    For each cam_id:
        crop[cam_id] = smooth_bbox[cam_id].to_crop(images[cam_id])

Step 5: Preprocess keypoint detection
    For each cam_id:
        kp_tensor[cam_id], kp_meta[cam_id] = keypoint_detector.preprocess(crop[cam_id])
    → Optionally vectorized if all crops share the model input size (they will, post-letterbox)

Step 6: Batched keypoint detection  [one GPU call]
    raw_kp = session.run_batched(keypoint_detector.model_name, kp_tensor)

Step 7: Postprocess + keypoint filtering per camera
    For each cam_id:
        keypoints[cam_id] = keypoint_detector.postprocess(raw_kp[cam_id], kp_meta[cam_id])
        keypoints[cam_id], states[cam_id] = keypoint_filter.smooth(keypoints[cam_id], states[cam_id])

Step 8: Child stages (per-camera, recursive)
    For each child_stage:
        child_obs[cam_id], child_states[cam_id] = child_stage.run_batch(crop, states, context)

Step 9: Assemble StageObservation per camera
```

A stage with no `ObjectDetector` skips steps 1–4 and uses the full image as the crop — only **one GPU call** total.

---

## Parallelizing Pre/Postprocessing

Preprocessing and postprocessing are NumPy-heavy and release the GIL. Three strategies apply, in order of preference:

### 1. Vectorization (preferred, zero overhead)

When all cameras share the same resolution, preprocessing can operate on a `(N, H, W, 3)` array rather than N separate arrays. Letterboxing, normalization, and transposing to `(N, 3, H, W)` all support this. Similarly, coordinate untransforms and visibility thresholding in postprocess can be batched over the N-axis.

This is the expected common case for synchronized multi-cam freemocap setups (all cameras typically use the same resolution).

### 2. ThreadPoolExecutor (heterogeneous resolutions)

When cameras differ in resolution, per-camera preprocessing cannot be vectorized. A `ThreadPoolExecutor` with one task per camera provides genuine parallelism because NumPy operations release the GIL:

```python
with ThreadPoolExecutor(max_workers=len(images)) as pool:
    futures = {cam_id: pool.submit(detector.preprocess, img)
               for cam_id, img in images.items()}
    preprocessed = {cam_id: f.result() for cam_id, f in futures.items()}
```

### 3. Pipeline parallelism for realtime (CPU and GPU)

For live camera streams, the three stages can be overlapped across consecutive frames using a producer-consumer pipeline:

```
Frame N-1:  [preprocess] → [GPU infer] → [postprocess + temporal]
Frame N:                   [preprocess] → [GPU infer] → [postprocess + temporal]
Frame N+1:                               [preprocess] → ...
```

Two queues connect the three stages. Each runs in its own thread. On CPU, where inference is slow (10–50 ms), preprocessing the next frame while inference runs on the current frame gives near-full CPU utilization without multiprocessing overhead.

This is especially valuable for CPU users who cannot batch across cameras as efficiently as CUDA.

---

## MediaPipe: Threading Fallback

MediaPipe does not expose tensor-level inference — it owns its entire preprocessing, inference, and postprocessing pipeline internally. `run_batched()` is not available.

For MediaPipe detectors, `DetectionStage.run_batch()` falls back to a `ThreadPoolExecutor` that calls `detect()` once per camera concurrently. Each camera gets its own `MediaPipeSession` (created at setup time — one per camera), and MediaPipe releases the GIL during `process()`, so the calls run in parallel:

```python
with ThreadPoolExecutor(max_workers=len(images)) as pool:
    futures = {cam_id: pool.submit(detector.detect, img)
               for cam_id, img in images.items()}
    results = {cam_id: f.result() for cam_id, f in futures.items()}
```

The N-camera speedup is bounded by MediaPipe's internal threading behaviour rather than GPU batch efficiency, but it is meaningfully faster than sequential processing.

---

## `process_folder` with Batching

Once `process_batch` is available, `process_folder` will be updated to:

1. Open all N video files simultaneously
2. Per frame: read one frame from each file → `dict[cam_id, NDArray]`
3. Call `tracker.process_batch(images, frame_number, states)`
4. Accumulate per-camera `DataStore`s
5. Save all N arrays at the end

This gives true batched GPU inference across all cameras for the entire recording, with the same single-session memory budget as single-camera processing.

---

## Implementation Checklist

### Core inference split — done

- [x] Define per-detector `Metadata` dataclasses (`YoloxMetadata`, `RTMPoseMetadata`, `EmptyMetadata`) — `core/detectors/metadata.py`
- [x] Add abstract `preprocess` / `postprocess` to `ObjectDetector` and `KeypointDetector` base classes
- [x] Implement `preprocess` / `postprocess` in `YoloxPersonDetector`
- [x] Implement `preprocess` / `postprocess` in `RTMPoseKeypointDetector` (wholebody, body, face, hand variants)
- [x] Implement `preprocess` / `postprocess` in `MediapipePoseDetector`, `MediapipeFaceDetector`, `MediapipeHandDetector` (trivial — batch path uses thread-pool `detect()` instead)
- [x] Implement `preprocess` / `postprocess` in `CharucoDetector`, `ArucoDetector`, `PrecomputedObjectDetector`
- [x] Add `OnnxSession.run_batched(model_name, tensors: dict[str, NDArray]) -> dict[str, list]` — stacks N `(3,H,W)` tensors, one ORT call, splits back by camera key using `out[i:i+1]` to preserve the batch dim
- [x] Add `DetectionStage.run_batch(images, states, context)` — ONNX path uses two-phase batched GPU calls; non-ONNX falls back to `ThreadPoolExecutor` calling `detect()` per camera
- [x] Add `Tracker.process_batch(images, frame_number, states, timestamp_ms)`

### Temporal state reset — done (not originally on checklist)

- [x] Add `reset_temporal_state()` no-op to `ObjectDetector` and `KeypointDetector` base classes
- [x] Override `reset_temporal_state()` in all three MediaPipe detectors — closes and recreates the landmarker so no timestamp or tracking state bleeds between videos
- [x] Propagate `reset_temporal_state()` through `DetectionStage` and `Tracker`
- [x] `process_folder` calls `tracker.reset_temporal_state()` between videos

### Batch video processing — done (not originally on checklist)

- [x] `process_video(tracker, annotator, input_path, output_dir, ...)` — single-video frame loop, saves `(frames, points, 3)` `.npy`
- [x] `process_folder(tracker, annotator, video_dir, output_dir, ...)` — batched across all cameras; opens all N videos simultaneously, one `process_batch()` call per frame

### Testing — done

- [x] `test_process_batch.py` — 7 tests covering `Tracker.process_batch()` with MediaPipe body (CPU, no GPU required): key/shape consistency, state isolation, single-camera equivalence
- [x] `test_run_batched.py` — 26 tests (require onnxruntime; auto-skip otherwise): ONNX `preprocess` tensor shapes/dtypes, `run_batched` N=1 matches `detect()`, N=2 key preservation and shape consistency, full `DetectionStage.run_batch()` YOLOX→RTMPose end-to-end
- [x] `test_process_video.py` — `TestProcessFolder` covers `process_folder`: all outputs produced, correct shape, keypoint count consistency across cameras, shapes match single-camera `process_video` path

### Packaging fix — done (not originally on checklist)

- [x] `pyproject.toml`: removed `onnxruntime` from `exclude-dependencies`; added `[[tool.uv.dependency-metadata]]` override for `rtmlib==0.0.14` to strip its transitive `onnxruntime` dep — fixes `uv sync --extra all-cpu` silently not installing onnxruntime

### `process_folder` batched rewrite — done

- [x] `process_folder` opens all N videos simultaneously and calls `tracker.process_batch()` once per frame
- [x] `OnnxSession.batch_size` stored as an instance field; `run_batched` warns when camera count ≠ `batch_size`
- [x] `DetectionStage._cam_kp_detectors` — per-camera detector instances for non-ONNX backends (e.g. MediaPipe VIDEO mode), lazily created in `run_batch` and cleared by `reset_temporal_state()`
- [x] `process_folder` calls `tracker.reset_temporal_state()` at the start so repeated calls don't inherit stale per-camera detector state

---

### Still to do

- [ ] **Pipeline parallelism for realtime** — the producer-consumer overlap described in the "Parallelizing Pre/Postprocessing" section (preprocess frame N while inferring on frame N-1) is not yet implemented. Relevant for CPU users in live streaming scenarios.
