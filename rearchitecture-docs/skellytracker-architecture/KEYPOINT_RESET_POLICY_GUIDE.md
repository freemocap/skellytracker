# Keypoint Reset Policy Guide

This guide explains the keypoint-detector reset policy: what problem it solves, and how to turn it on for your tracking pipeline.

## The problem

MediaPipe's `PoseLandmarker` (and similar landmarkers) in `VIDEO` running mode use a track-then-detect pipeline. After the first successful detection, later frames run a faster tracking step instead of full re-detection. If tracking is lost — occlusion, fast motion, a dropped frame, confidence dipping below `min_pose_presence_confidence` — the landmarker can return an **empty result on every subsequent frame**, even when the subject is clearly visible. Re-detection is not automatically retried; the pipeline needs to be reset from the outside to recover.

Left alone, this shows up as a camera that silently stops producing keypoints partway through a recording, while other cameras (or the same tracker in `IMAGE` mode) keep working fine.

## The fix: `KeypointResetPolicyConfig`

Every `DetectionStageConfig` accepts a `keypoint_reset_policy` field:

```python
from skellytracker.core.temporal_processing.temporal_processing_config import (
    KeypointResetPolicyConfig,
)

policy = KeypointResetPolicyConfig(max_consecutive_misses=10)
```

Each keypoint detector in the stage now tracks how many **consecutive frames** it returned zero valid keypoints (`n_valid == 0`), checked *before* any confidence/visibility filtering. Once that streak reaches `max_consecutive_misses`, the stage calls `detector.reset_temporal_state()` on that detector and the counter resets to `0`. For MediaPipe, `reset_temporal_state()` closes and recreates the landmarker, discarding the stale internal tracking state and forcing a full detection on the next frame.

A "miss" is strictly "the detector found nobody." A frame where the pose was found but a limb has low visibility (e.g. out of frame, occluded) does **not** count as a miss — that's normal, expected tracking behavior, not stuck state.

The default is `max_consecutive_misses=None`, which disables the policy entirely — no behavior change unless you opt in.

## Enabling it on a stage

Pass the policy when building the `DetectionStageConfig` for your MediaPipe stage:

```python
from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.temporal_processing.temporal_processing_config import (
    KeypointResetPolicyConfig,
)
from skellytracker.core.detectors.keypoint_detectors.mediapipe.body.mediapipe_pose_detector import (
    MediapipePoseDetectorConfig,
)

stage_config = DetectionStageConfig(
    name="body",
    keypoint_detectors=[MediapipePoseDetectorConfig()],
    keypoint_reset_policy=KeypointResetPolicyConfig(max_consecutive_misses=10),
)
```

This `stage_config` then goes into your `TrackerConfig` / `Tracker.create(...)` as usual — no other wiring is required. The policy applies independently to every keypoint detector in the stage (and, in `process_video_list`'s batched multi-camera path, independently per camera), since each stateful detector instance tracks its own miss streak.

## Choosing a threshold

`max_consecutive_misses` is in **frames**, not milliseconds, so scale it to your frame rate:

| FPS | Suggested threshold | Reset latency |
|---|---|---|
| 30 | 8–10 | ~270–330 ms |
| 60 | 15–20 | ~250–330 ms |
| 15 | 5–8 | ~330–530 ms |

Lower thresholds recover faster from a stuck tracker but risk resetting on transient single-frame misses (e.g. a brief motion blur). Start around 10 frames at 30 fps and adjust based on how often your subject leaves the frame or gets occluded.

## What this does *not* fix

- **Low-confidence keypoints.** If landmarks are detected but below your visibility threshold, that's not a miss and won't trigger a reset. Tune `min_pose_detection_confidence` / `min_pose_presence_confidence` on the detector config for that, not this policy.
- **Stateless detectors.** ONNX-backed detectors (RTMPose, CompositeGPU's RTMO/RTMPose submodels) run full detection every frame already — there's no internal tracking state to get stuck, so `reset_temporal_state()` is a no-op for them regardless of this policy's setting.
- **Bounding-box tracking loss.** If your stage also has an `ObjectDetector` (e.g. YOLOX) with a `BBoxPolicyConfig`, that's a separate, already-existing mechanism (`bbox_policy.fitness_checks`, e.g. `MaxFramesWithoutRedetectConfig`) for re-running detection when the *bbox* looks stale. The two policies are complementary and can both be set on the same stage.

## How it works internally

- `StageState.consecutive_misses: list[int]` holds one counter per keypoint detector (per camera, in the batched path).
- `DetectionStage.run()` / `DetectionStage.run_batch()` increment or reset that counter each frame based on `Keypoints.n_valid`, before any keypoint smoothing is applied (smoothing can gap-fill NaNs, which would otherwise mask a miss).
- `KeypointResetPolicy.should_reset(consecutive_misses)` (in `skellytracker/core/temporal_processing/keypoint_reset_policy.py`) decides whether the streak has crossed the configured threshold.
- On trigger, `detector.reset_temporal_state()` is called on that specific detector instance — for MediaPipe, this closes and recreates the underlying `PoseLandmarker`/`HandLandmarker`/`FaceLandmarker`.

If you're adding a new stateful keypoint detector and want it to support this policy, implement `reset_temporal_state()` on it (see `MediapipePoseKeypointDetector.reset_temporal_state` for reference) — the stage-level plumbing is already generic.
