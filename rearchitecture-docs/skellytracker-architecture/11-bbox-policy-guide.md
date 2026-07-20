# BBox Policy — Current API and freemocap Integration Guide

This documents the **as-built** `BBoxPolicy` / bbox-smoothing API (`skellytracker/core/temporal_processing/`, `skellytracker/core/tracker/detection_stage.py`, `skellytracker/core/tracker/tracker_state.py`), and what a caller building an RTMPose tracker — freemocap in particular — needs to configure to get correct, non-collapsing bbox tracking.

> **Note:** [`09-temporal-processing.md`](./09-temporal-processing.md) is a pre-implementation design doc and is out of date on this topic (it describes a `smooth_center`/`smooth_size` EMA state and a simpler `BBoxPolicy` that no longer match the code). This doc reflects what's actually in the repo today.

## Why this exists

Running the YOLOX person detector every frame is expensive. `BBoxPolicy` lets a `DetectionStage` skip it most frames and instead derive the crop for the keypoint detector (RTMPose) from the *previous* frame's keypoints — re-running YOLOX only periodically or when the track looks unreliable. Getting this wrong doesn't just waste compute — it directly causes keypoint jitter (bbox flapping between shapes) or a runaway collapse (the crop shrinks to nothing, which crashes `cv2.warpAffine` on a zero-size image). Both failure modes were hit and fixed during this session; this guide captures the resulting design so it isn't re-broken by a naive freemocap integration.

## The API surface

### Config (`temporal_processing_config.py`)

```python
class BBoxPolicyConfig(BaseModel):
    redetect_interval: int = 1                    # frames between forced YOLOX runs
    fitness_checks: list[BBoxFitnessCheckConfig] = []
    keypoint_bbox_expansion: float | None = None  # enables keypoint-derived crop on skip frames
    keypoint_bbox_min_visibility: float = 0.0
    min_shrink_ratio_per_frame: float | None = 0.999   # per-frame shrink-rate clamp
    min_detected_bbox_ratio: float | None = 0.5        # floor vs. last actual YOLOX box
    min_bbox_size_px: float = 80.0                    # absolute floor, prevents zero-size crop

class BBoxSmoothingConfig(BaseModel):
    alpha: float = 0.5   # EMA weight on the *previous* smoothed crop; higher = more lag
```

`BBoxFitnessCheckConfig` is a discriminated union (`kind` field) of:

| Config | Fires when |
|---|---|
| `MinKeypointVisibilityConfig(threshold)` | mean visibility of valid keypoints < threshold |
| `KeypointsWithinBBoxRatioConfig(threshold)` | fewer than `threshold` fraction of keypoints fall inside the current crop |
| `MaxFramesWithoutRedetectConfig(n_frames)` | absolute safety cap, redundant with `redetect_interval` unless set tighter |
| `BBoxAreaCollapseConfig(min_area_ratio, expansion_ratio, min_visibility)` | this frame's keypoint-derived box is `< min_area_ratio` of the current crop's area — a rare, genuine-track-loss safety net, **not** the mechanism that prevents routine shrink (see below) |

Attach both to a stage:

```python
DetectionStageConfig(
    name="wholebody",
    object_detector=YoloxPersonDetectorConfig(),
    keypoint_detectors=[RTMPoseDetectorConfig(model_name="rtmw-x-l_256x192")],
    bbox_policy=BBoxPolicyConfig(...),
    bbox_smoothing=BBoxSmoothingConfig(alpha=0.4),
)
```

### State (`tracker_state.py`)

```python
class BBoxSmoothingState:
    smooth_bbox: BoundingBox | None = None          # the actual crop used this frame (post-EMA)
    last_detection_frame: int | None = None         # frame_number of the last YOLOX run
    keypoint_tracked_bbox: BoundingBox | None = None  # tight-around-keypoints + 1 expansion, every frame
    last_detected_bbox: BoundingBox | None = None     # raw box from the last actual YOLOX run (unsmoothed)
```

`StageState` (one per `DetectionStage`, keyed by stage name inside `TrackerState.stage_states`) carries `bbox_state` plus `last_keypoints`, per-keypoint smoothing states, and `consecutive_misses`. **This must persist across frames for the same camera** — see "State lifetime" below.

### Observation (`observation.py`)

```python
class StageObservation:
    bounding_boxes: list[BoundingBox]
    keypoints: Keypoints | None
    detector_ran: bool   # True iff YOLOX actually ran this frame (not reused/predicted)
```

`detector_ran` is purely informational (annotator debug coloring), not consumed by the policy itself.

## Execution order (per stage, per frame)

```
DetectionStage.run(image, state, context):
  1. should_redetect(frame_number, state)?
       True  → run ObjectDetector(YOLOX) → raw_bbox; reset last_detection_frame
       False → predict_bbox(state) → raw_bbox (see below)
  2. EMA-smooth raw_bbox against state.bbox_state.smooth_bbox (if bbox_smoothing set)
       → smoothed_bbox becomes this frame's crop
  3. crop image to smoothed_bbox, run keypoint detector(s), translate back to full-frame coords
  4. refresh state.bbox_state.keypoint_tracked_bbox from *this frame's* keypoints
     — unconditionally, whether step 1 redetected or not
  5. run child stages, assemble StageObservation
```

Step 4 running unconditionally every frame — not just on skip frames — is what makes `predict_bbox` have fresh input to work with next frame.

## `predict_bbox` — the part that actually matters

```python
def predict_bbox(self, stage_state: StageState) -> BoundingBox | None:
    tracked = stage_state.bbox_state.keypoint_tracked_bbox
    if self.keypoint_bbox_expansion is None or tracked is None:
        return stage_state.bbox_state.smooth_bbox        # keypoint tracking disabled

    candidate = tracked.scaled(1.0 + 2.0 * self.keypoint_bbox_expansion)  # 2nd expansion

    prev_crop = stage_state.bbox_state.smooth_bbox
    if prev_crop is not None and self.min_shrink_ratio_per_frame is not None:
        # clamp: candidate can't shrink faster than min_shrink_ratio_per_frame
        # relative to *last frame's actual crop*
        ...
    detected = stage_state.bbox_state.last_detected_bbox
    if detected is not None and self.min_detected_bbox_ratio is not None:
        # floor: candidate can't shrink below min_detected_bbox_ratio of the
        # last actual YOLOX box, regardless of how long it's been since redetect
        ...
    # absolute floor: candidate width/height >= min_bbox_size_px
    ...
    return candidate
```

Three things to understand before configuring this for a new tracker:

1. **Double expansion.** `keypoint_tracked_bbox` (tight box around this frame's keypoints, expanded once) is expanded *again* here before being handed back as next frame's crop. This mirrors `skellytracker/old/rtmpose_tracker`'s update-then-predict two-stage expansion. A single expansion is not enough margin: keypoints sitting near the crop edge (or with a one-frame confidence dip — RTMPose SIMCC scores are noisy near the visibility threshold) fall outside the *next* crop and can never be re-acquired before the next scheduled redetect.

2. **Expansion alone cannot prevent collapse — only a shrink-rate clamp can.** Relative (percentage) expansion of a tight box still yields a *smaller* absolute box than the previous crop whenever this frame detects fewer keypoints than last frame (e.g. a hand or shoulder becomes unreliable near the crop boundary). There is no stable equilibrium in that recurrence — only a slower or faster collapse toward zero. `min_shrink_ratio_per_frame` (default `0.999`, i.e. the crop can lose at most 0.1% of its size per frame) is compared against the *previous actual crop*, not the shrinking tight box, which directly breaks the feedback loop instead of merely slowing it.

3. **A per-frame rate limit alone still drifts to the tight keypoint box over a long enough `redetect_interval`** — it only paces the shrink, it doesn't bound the total amount. `min_detected_bbox_ratio` (default `0.5`) adds a second floor tied to the object detector's last actual measurement: the crop can never shrink below `min_detected_bbox_ratio` of YOLOX's last real box, no matter how many frames have elapsed since redetect or how much keypoint visibility has degraded. This matters specifically for the case of a keypoint sitting at the *frame edge* (e.g. a shoulder when the subject is near the camera's field-of-view boundary) — a keypoint that's briefly undetected there would otherwise cause the tracked-box to contract away from that region permanently until the next redetect, since the keypoint detector never sees outside its crop. `min_bbox_size_px` is a last-resort absolute floor so a degenerate crop can never reach `cv2.warpAffine`.

Do **not** rely on `BBoxAreaCollapseConfig` as a substitute for the shrink-rate clamp — it's a coarse "give up and redetect" safety net for genuine track loss (person leaves frame, track jumps to someone else), and if the shrink-rate clamp is disabled or misconfigured it will fire almost every frame instead of preventing the underlying problem (this was tried and reverted during development — see git history on this file for the exact symptom).

## What freemocap needs to change to build a correct RTMPose tracker

freemocap processes **synchronized recorded video files**, not a live webcam — this matters, because it means the real fps is known exactly (`cv2.CAP_PROP_FPS`), unlike a live camera stream. Use it.

```python
from skellytracker.core import DetectionStageConfig, Tracker, TrackerConfig
from skellytracker.core.detectors.keypoint_detectors.rtmpose import RTMPoseDetectorConfig
from skellytracker.core.detectors.object_detectors.yolox import YoloxPersonDetectorConfig
from skellytracker.core.temporal_processing.temporal_processing_config import (
    BBoxPolicyConfig, BBoxSmoothingConfig, KeypointsWithinBBoxRatioConfig,
)

fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
redetect_seconds = 5.0  # matches old default cadence

stage = DetectionStageConfig(
    name="wholebody",
    object_detector=YoloxPersonDetectorConfig(),
    keypoint_detectors=[RTMPoseDetectorConfig(model_name="rtmw-x-l_256x192")],
    bbox_policy=BBoxPolicyConfig(
        redetect_interval=max(1, round(redetect_seconds * fps)),   # use REAL fps, not an assumed one
        keypoint_bbox_expansion=0.05,        # matches old's tracking_expansion_ratio
        fitness_checks=[KeypointsWithinBBoxRatioConfig(threshold=0.5)],
        min_shrink_ratio_per_frame=0.999,     # leave at default unless you see specific issues
        min_detected_bbox_ratio=0.5,         # leave at default unless you see specific issues
        min_bbox_size_px=80.0,               # see multi-camera note below
    ),
    bbox_smoothing=BBoxSmoothingConfig(alpha=0.4),
)
```

Key points, in order of how likely they are to bite:

1. **`redetect_interval` is frame-count based, not wall-clock.** There is no fps auto-detection inside `BBoxPolicy` — it only ever sees `frame_number`. freemocap must compute `redetect_interval = round(redetect_seconds * actual_video_fps)` itself, per recording, using the real fps from the video file (this is *more* accurate than what our live-webcam demo does, which has to assume ~30fps since it doesn't know the camera's true rate up front).

2. **`min_bbox_size_px` is an absolute pixel value, not resolution-relative.** If freemocap's cameras record at different resolutions (or much higher/lower than typical webcam resolution), `80.0` px may be too small (barely bigger than a face at 4K) or too restrictive (a large fraction of frame at low res). Scale it relative to the actual video resolution, e.g. `min_bbox_size_px = min(width, height) * 0.05`.

3. **State must persist per camera across the whole recording, not be recreated per frame.** `TrackerState`/`StageState` carry `bbox_state.keypoint_tracked_bbox` and `last_detection_frame` — if you build a fresh `TrackerState()` every frame (easy mistake when adapting single-shot-per-frame code), `should_redetect` will always see `last_detection_frame is None` and redetect every single frame, silently disabling the whole skip-frame mechanism. Call `tracker.reset_temporal_state()` once per new recording session, then thread the same `TrackerState` (or per-camera dict of them, for `process_batch`) through every frame — this is exactly what `process_video_list`/`process_folder` in `skellytracker/core/io/process_video.py` already do; prefer calling those over hand-rolling the loop.

4. **Multi-camera batched redetection is synchronized, not independent.** `DetectionStage.run_batch` (used by `Tracker.process_batch`, which `process_video_list` calls) redetects **all** cameras together if *any* camera's `should_redetect` fires, to keep ONNX batch size at 0 or N (never a ragged batch, which would force CoreML/TensorRT recompilation). This means one camera losing track can trigger a full-batch YOLOX run slightly ahead of schedule for the others — harmless, but worth knowing if you're trying to reason about exact per-camera redetect cadence from logs.

5. **`keypoint_bbox_expansion=None` disables all of this** (both the tracked-bbox refresh and `predict_bbox`'s keypoint path) and falls back to simply reusing the last crop verbatim on skip frames — fine for a mostly-static subject, but won't track a moving person between YOLOX runs. Make sure it's set to a non-`None` value if freemocap wants tracking between redetects at all — this is not the default (`BBoxPolicyConfig.keypoint_bbox_expansion` defaults to `None`).

6. **For debugging cadence during integration**, use `StageAnnotationSchema(draw_boxes=True)` on the annotator — it colors the box green when `detector_ran` (YOLOX actually ran) and orange when the crop was carried over from the keypoint-tracked prediction (`keypoint_annotator.py`, `box_color_detected` / `box_color_reused`). This was built specifically to diagnose the bugs this doc describes and is the fastest way to confirm a new tracker config is redetecting on the cadence you expect, rather than every frame or never.

## Backward compatibility

All new fields (`BBoxPolicyConfig.min_shrink_ratio_per_frame`, `min_detected_bbox_ratio`, `min_bbox_size_px`; `BBoxSmoothingState.keypoint_tracked_bbox`, `last_detected_bbox`) have defaults, so existing configs that only set `redetect_interval`/`fitness_checks` continue to work unchanged and get the shrink protection for free. The one place this could bite: if freemocap ever serializes a `TrackerState`/`StageState` to disk to resume a paused session (e.g. `pickle`), a state saved before this change will be missing `keypoint_tracked_bbox`/`last_detected_bbox` on load — reconstruct them as `None` on migration (they self-heal within one frame/one redetect cycle respectively once frames start flowing again).
