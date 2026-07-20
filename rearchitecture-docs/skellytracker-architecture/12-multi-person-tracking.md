# Multi-Person Tracking (Single Camera)

Cross-frame identity tracking for multiple people within one camera's video stream: given N people detected per frame, assign each a stable `track_id` that persists across frames. This is distinct from — and sits upstream of — the *cross-camera* correspondence problem (matching person A in camera 0 to person A in camera 1, used for triangulation), which is out of scope for skellytracker and lives in freemocap.

## Why It Exists

`Tracker` / `DetectionStage` (see [01-tracker.md](./01-tracker.md), [03-detection-stage.md](./03-detection-stage.md)) are single-subject by construction: `Observation` holds exactly one person, and the object detector is capped to its top detection (`YoloxPersonDetectorConfig.max_detections=1` by default). Multi-person tracking needed a parallel orchestrator that:

1. Discovers *all* people in a frame, not just the best one.
2. Keeps each person's identity consistent frame to frame (a standard multi-object-tracking problem — the single-camera analog of SORT/DeepSORT), while reusing all of the existing single-person temporal-smoothing machinery ([09-temporal-processing.md](./09-temporal-processing.md)) *per track* instead of per frame.

## What Was Built

| Component | File | Role |
|---|---|---|
| `MultiPersonTrackingConfig` | `core/temporal_processing/multi_person_config.py` | IoU/keypoint cost weights, match-cost gate, `max_age`, `min_hits` |
| `track_association` | `core/temporal_processing/track_association.py` | `iou()`, cost matrices, `associate()` — Hungarian assignment via `scipy.optimize.linear_sum_assignment` |
| `PersonTrackState` | `core/tracker/person_track.py` | Per-track identity + the same `StageState` a single-person `Tracker` uses |
| `DetectionStage.detect_raw_at_bbox()` / `.finalize_track()` | `core/tracker/detection_stage.py` | Split the existing `run()` pipeline into "detect raw keypoints at an external bbox" and "apply this track's temporal smoothing" so candidate probing and match commitment don't re-run inference |
| `MultiPersonTracker` | `core/tracker/multi_person_tracker.py` | Orchestrator: propose → detect → associate → finalize → age out |
| `MultiPersonObservation` / `MultiPersonDataStore` | `core/data_primitives/multi_person_observation.py`, `data_store.py` | Per-frame multi-person output and storage (wraps `Observation`/`DataStore`, no changes to either) |

### Per-frame algorithm (`MultiPersonTracker.process_image`)

```
1. Run the root stage's ObjectDetector on the whole frame
     → candidate bboxes (every person, not just the best one)

2. For each candidate, DetectionStage.detect_raw_at_bbox():
     crop → run keypoint detectors → raw (unsmoothed) keypoints
     (passes a fresh, empty StageState — no track identity yet)

3. track_association.associate(existing tracks, candidates):
     cost = weighted blend of (1 - IoU) and normalized keypoint displacement
     Hungarian assignment, gated by max_match_cost
     → matches, unmatched_tracks, unmatched_detections

4. Matched candidates: DetectionStage.finalize_track()
     reuses the keypoints from step 2 (no repeat inference) and applies
     *that track's own* bbox EMA / keypoint filter / reset-policy state —
     identical math to single-person Tracker.process_image, keyed per track

5. Unmatched detections → spawn a new PersonTrackState
   Unmatched tracks     → time_since_update += 1; dropped once > max_age

6. Emit MultiPersonObservation: confirmed tracks (hits >= min_hits)
   that matched this frame
```

The key structural move is splitting `DetectionStage.run()` (single-person, object-detector-driven) into reusable pieces — `_detect_and_translate`, `_apply_reset_policy`, `_smooth_bbox`, `_smooth_keypoints`, `_run_children` — so `run()` itself is unchanged in behavior (verified: all 250 pre-existing tests pass unmodified) while `detect_raw_at_bbox`/`finalize_track` recombine the same pieces around an *externally supplied* bbox instead of the stage's own object detector.

### Association cost

`combined_cost_matrix()` blends two signals, each normalized to `[0, 1]`:

- **IoU cost** = `1 - iou(track_bbox, candidate_bbox)`
- **Keypoint cost** = mean pixel displacement between the track's last keypoints and the candidate's raw keypoints (shared, confident point names only), clipped to `[0, 1]` after dividing by a normalization constant

A pair with only one finite signal (e.g. a brand-new track with no keypoint history yet) falls back to that signal alone rather than being penalized for the missing one. Pairs costing more than `max_match_cost` are gated out even if the Hungarian solver would otherwise pick them — this is what lets a track legitimately die (person left frame) instead of snapping onto an unrelated detection.

### Config knobs (`MultiPersonTrackingConfig`)

| Field | Default | Effect |
|---|---|---|
| `iou_weight` / `keypoint_weight` | 0.5 / 0.5 | Blend ratio for the cost matrix |
| `max_match_cost` | 0.8 | Gate — pairs above this never match |
| `max_age` | 10 | Frames a track survives unmatched before being dropped |
| `min_hits` | 3 | Matched frames required before a track is "confirmed" and emitted (suppresses one-frame detector false positives) |

## What's Not Done — Remaining Work for "Full" Multi-Person Tracking

These are known, documented gaps, not oversights — each is called out in code comments at the relevant spot.

1. **No re-identification across a tracking gap.** A person who leaves frame for longer than `max_age` and returns gets a *new* `track_id`, not their old one. Real re-ID would need an appearance embedding (e.g. a small ReID CNN or even a color-histogram descriptor) compared against recently-dropped tracks before spawning a new ID — meaningfully more machinery (an embedding model, a gallery of recent tracks, a similarity threshold) than the geometric IoU/keypoint cost used here.

2. **Stateful keypoint detectors aren't correctly per-track.** `detect_raw_at_bbox()` runs candidate probing against a fresh, empty `StageState`, so a stateful backend (MediaPipe's VIDEO-mode tracking) never sees a consistent temporal stream for a given physical person — it's effectively reset every frame during candidate detection. This is fine for stateless ONNX detectors (RTMPose), which is the intended default for multi-person mode, but multi-person + MediaPipe would need one detector *instance* per track (mirroring what `DetectionStage.run_batch` already does per-camera via `_cam_kp_detectors`), allocated/retired alongside track birth/death.

3. **No batched multi-camera + multi-person combination.** `MultiPersonTracker` only handles one camera; `Tracker.process_batch`/`DetectionStage.run_batch` handle N cameras but stay single-person. Combining both (N cameras × M people, batched ONNX calls) hasn't been attempted — it would need the candidate/associate/finalize loop to happen per camera, with the ONNX batching trick applied across the *flattened* candidate set per frame rather than one call per camera.

4. **Association cost is simple by design.** IoU + mean keypoint displacement is standard SORT-tier and works well for well-separated people with steady motion. It will struggle with:
   - **Heavy occlusion / crossing paths at close range** — no motion model (Kalman velocity prediction) is used to disambiguate; the association only looks at the current frame's raw geometry vs. last frame's committed result.
   - **Crowded scenes** — cost matrix is `O(tracks × detections)`, fine at normal scale, but no NMS-style suppression exists between *candidates themselves* beyond what the object detector's own NMS already does.
   A constant-velocity motion predictor (extrapolate `track.last_bbox` forward before computing IoU, similar to classic SORT's Kalman step) would likely be the single highest-value next addition here.

5. **No dedicated multi-person `Annotator`.** The example script (`examples/run_multiperson_on_video.py`) draws boxes/IDs directly with `cv2` rather than through the `Annotator` abstraction ([07-supporting-objects.md](./07-supporting-objects.md)) that single-person tracking uses — a `MultiPersonAnnotator` wrapping the existing per-stage annotator per track would be a small, mechanical follow-up.

6. **`max_persons` / candidate volume is unbounded by policy.** `YoloxPersonDetectorConfig.max_detections` caps candidates per frame, but there's no cost-based pruning of implausible candidates (e.g. tiny boxes, boxes at frame edges) before they enter the cost matrix — currently whatever passes the object detector's own confidence/NMS threshold is a candidate.

7. **Downstream: cross-camera correspondence remains someone else's problem, by design.** Once multi-person, single-camera tracks are stable, matching those tracks across synchronized cameras (for triangulation) is explicitly out of scope for skellytracker — see the EasyMocap-style affinity/SVT approach referenced in the original design discussion. Nothing here precludes wiring `MultiPersonObservation` per camera into that kind of downstream matcher; it just isn't part of this module.

## Tests

- `skellytracker/tests/test_track_association.py` — `iou`, cost matrices, `associate()` (matching, gating, empty-track/empty-detection edge cases)
- `skellytracker/tests/test_multi_person_tracker.py` — full `MultiPersonTracker` lifecycle against scripted (no-ONNX) detections: stable IDs across frames, track drop after `max_age` + new ID on reappearance, non-overlapping crossing paths stay correctly assigned

Both use hand-written `ObjectDetector`/`KeypointDetector` test doubles (`ScriptedObjectDetector`, `CenterPointKeypointDetector`) rather than real models, so they run fast and deterministically without onnxruntime.
