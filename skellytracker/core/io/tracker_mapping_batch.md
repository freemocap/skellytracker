# Batched tracker-to-landmark mapping

`tracker_mapping_batch.map_tracker_batch(mapping, tracker_names=names,
positions=points)` accepts a `TrackerMapping` or `CompositeTrackerMapping` and
float-compatible `(frames, keypoints, 3)` positions. It returns
`MappedLandmarkBatch(landmark_names, positions, present)`, with positions shaped
`(frames, landmarks, 3)` and a boolean `(frames, landmarks)` presence mask.
Any nonfinite input XYZ marks that whole input point absent. Consumers must use
the presence mask, not interpret missing observations as zeros.

The implementation reads validated mapping snapshots. Affine mapping definitions
are grouped by source count and reduced across both frames and landmarks. Means
require every source; weighted mappings preserve left-to-right accumulation.
Anatomical offsets compute directions, reference lengths, orthogonal bases and
cross products over frame arrays. Definition dispatch still loops over the small
set of distinct anatomical formulas; it never calls scalar mapping per frame.
Missing-source and degenerate-axis rules match the existing scalar mapping.
Offsets read only raw and pass-one affine observations, as in the scalar API.

Call with bounded frame chunks for very large recordings. Mapping has no temporal
state; chunk boundaries do not affect results. The API does not map confidence
scores. Existing scalar and quality APIs are unchanged.

## Validation and measurements — 2026-10-12

The new tests compare outputs and absence masks against scalar mapping for all
four authored RTMPose/MediaPipe body and hand definitions, missing frames/points,
zero geometry, prefix/composite/pass-through mapping, negative weights, and empty
inputs. The focused mapping suite passed 50 tests. Numeric parity uses
`rtol=1e-12, atol=1e-10`; exact floating-point equality is not promised.

Synthetic named coordinates with periodic missing observations, three repetitions,
alternating scalar/batch order, median wall time (including scalar dictionary
construction and batch snapshot restoration):

| Body mapping | Frames | Scalar s | Batch s | Speedup |
| --- | ---: | ---: | ---: | ---: |
| RTMPose | 222 | 0.2757 | 0.0044 | 62.6x |
| RTMPose | 1108 | 2.0335 | 0.0165 | 123.1x |
| MediaPipe | 222 | 0.4423 | 0.0093 | 47.7x |
| MediaPipe | 1108 | 2.1524 | 0.0161 | 133.8x |

These are mapping microbenchmarks, not actual test/sample recording timings or
whole-pipeline speedups. Local diagnostic script and raw timings are ignored at
`.test-artifacts/batch-mapping/benchmark.py` and `timing.json`.

## Consumer handoff

### Single-frame follow-up

The initial unprepared batch API was **not** a faster replacement for realtime scalar
mapping. A synthetic single-frame benchmark used 120 varying frames per repeat,
five repeats with rotating execution order, and periodic missing keypoints.
Both versions returned the existing landmark dictionary shape. The scalar input
dictionaries were prepared outside timing; the batch adapter includes converting
those dictionaries to arrays, snapshot restoration, mapping and conversion back.
Parity passed at `rtol=1e-12, atol=1e-10` for every tested frame and absence mask.

| Mapping | Scalar median ms/frame | Batch adapter median ms/frame | Added time ms/frame |
| --- | ---: | ---: | ---: |
| RTMPose body | 1.965 | 3.452 | 1.488 |
| RTMPose hand | 0.913 | 2.112 | 1.199 |
| MediaPipe body | 1.742 | 3.907 | 2.165 |
| MediaPipe hand | 0.979 | 2.374 | 1.394 |

Even supplying an already-built single-frame array was slower (respectively
3.225, 2.183, 3.432, 2.576 ms including output dictionary construction). These are
mapping-only microbenchmarks, not live-camera measurements; individual repeats
varied. No realtime call site was changed. A future low-latency implementation
should prepare and reuse mapping indices/definitions and batch anatomical offset
definitions together before testing again. The current implementation restores
snapshots and builds temporary arrays per call; no profiler attribution of their
individual costs was made here.

Reproduction and raw samples: ignored local files
`.test-artifacts/batch-mapping/single_frame.py` and `single-frame.json`.

### Publication

Publish SkellyTracker on `development-streaming` before FreeMoCap consumes this
API. Core must then refresh its pinned dependency through its normal integration
workflow, replace per-frame mapping/packing in fitting and reconstruction, and
compare actual reference-recording outputs. Do not use sibling imports or local
dependency overrides. No FreeMoCap integration change is included here.


## Prepared mapping follow-up

`PreparedTrackerMapping.prepare(mapping, tracker_names=names)` now compiles
source gathers, affine groups, fallback indices, anatomical mean groups, axis
selections and offset ratios once. Its `apply(tracker_names=names,
positions=points)` accepts either one frame `(1,K,3)` or multiple frames. It
batches anatomical offsets across landmarks as well as frames. The one-shot
`map_tracker_batch` delegates to this implementation, so there is no second
batch math implementation to maintain.

The plan snapshots configuration and owns read-only definition arrays. It stores
no coordinates, output arrays or mutable scratch buffers. Results are owned by
the caller and survive later calls. A configuration change requires explicit
preparation at the configuration boundary; a changed ordered keypoint axis is
rejected by `apply`. Coordinates and missing-point masks require no preparation.
The fingerprint identifies the prepared definition/axis and is calculated once,
not every frame. It is not an automatic watcher of mutable configuration.

Example:

```python
from skellytracker.core.io.prepared_tracker_mapping import PreparedTrackerMapping

# At setup/configuration change:
plan = PreparedTrackerMapping.prepare(mapping, tracker_names=names)
# Each frame; points has shape (1, len(names), 3):
landmarks = plan.apply(tracker_names=names, positions=points)
```

New benchmark, same synthetic protocol, five rotating-order repeats; median
milliseconds per frame including dictionary input/output adaptation:

| Mapping | Scalar | Prepared | Saved per frame |
| --- | ---: | ---: | ---: |
| RTMPose body | 2.361 | 0.353 | 2.008 |
| RTMPose hand | 1.358 | 0.320 | 1.038 |
| MediaPipe body | 2.602 | 0.443 | 2.158 |
| MediaPipe hand | 1.421 | 0.293 | 1.128 |

Use these contemporaneous pairs, not scalar times from the earlier experiment;
wall-clock timing varied between runs. No claim of live application latency or
wholebody speedup follows from these individual mapping tests. One-time setup is
excluded from these steady-state times.

52 focused mapping tests passed, including scalar numerical/missingness parity,
plan reuse without rereading definitions, changed configuration fingerprints,
axis mismatch rejection, missing-point recovery, empty input batches and output
ownership. No realtime or FreeMoCap call site has been changed. Consumer
integration remains behind the SkellyTracker publication handoff above.
Evidence: `.test-artifacts/batch-mapping/prepared_single_frame.py` and
`prepared-single-frame.json`.


## Completed dependency checklist

- [x] Prepared single-frame dictionary API.
- [x] Configuration, detector, ordered-axis and board-size replacement tests.
- [x] Batch confidence propagation.
- [x] Complete body-plus-hands benchmark and setup cost.
- [x] Pass-through benchmark.
- [x] Chunking and concurrent reuse tests.
- [x] API documentation and focused mapping regression suite.

### Consumer APIs and lifecycle

```python
# Once per mapping configuration and ordered keypoint layout:
plan = PreparedTrackerMapping.prepare(mapping, tracker_names=names)

# Existing single-frame named-point interface:
landmarks = plan.apply_frame(tracker_positions=frame_points)

# Array inputs, with optional conservative confidence propagation:
result = plan.apply_with_quality(
    tracker_names=names, positions=frame_arrays, quality=confidence_arrays,
)
```

`apply_frame` accepts a subset of the prepared keys; dictionary ordering does not
matter. Unknown keys fail explicitly so callers cannot silently reuse the wrong
layout. It packs to the prepared axis and returns finite mapped points only.
This adapter necessarily iterates dictionary entries at the object boundary;
the mapping calculations themselves operate on arrays.

`apply_with_quality` accepts `(F,K)` quality alongside `(F,K,3)` positions.
Present source scores must be finite and in [0,1]. Scores for absent positions
are ignored. Absent output scores are NaN. Each present mapped score is the
minimum of its required source scores, matching scalar `apply_with_quality`.
It returns `MappedLandmarkQualityBatch` with names, positions, presence and quality.
No scalar quality API is replaced implicitly.

Replace the plan when the detector/mapping/board configuration changes; do not
prepare or hash configuration every frame. A fingerprint identifies the prepared
snapshot but does not observe arbitrary subsequent mutation. No global cache or
shared mutable output buffers are used. Preparation stays at the owning session's
configuration boundary, and normal garbage collection releases superseded plans.
This implementation is independently usable by SkellyTracker clients and imports
neither FreeMoCap nor another subproject.

### Complete mapping benchmark

Synthetic changing coordinates and periodic occlusions, 100 frames per repeat,
five rotating-order repeats. Median ms/frame; preparation median over ten calls.
The dictionary comparison includes input packing and output dictionary creation.

| Mapping | Scalar dictionary | Prepared dictionary | Prepared array output | Setup ms |
| --- | ---: | ---: | ---: | ---: |
| RTMPose body + hands | 3.728 | 0.742 | 0.576 | 2.308 |
| MediaPipe body + hands | 3.788 | 0.752 | 0.520 | 2.748 |
| Pass-through 36 points | 0.033 | 0.086 | 0.029 | 0.211 |
| Pass-through 92 points | 0.067 | 0.127 | 0.031 | 0.366 |
| Pass-through 478 points | 0.440 | 0.504 | 0.067 | 1.339 |

Array output avoids dictionary conversion and is a different interface, not a
like-for-like replacement benchmark. Keep direct pass-through for dictionary
consumers: the prepared adapter has no demonstrated speed benefit there.
Synthetic marker counts represent simple board/dense point workloads; these are
not ChArUco detector or face neural inference benchmarks. Actual RTMPose and
MediaPipe schema names were used for the body-plus-hands cases. No live pipeline
was modified or benchmarked in this dependency stage.

Local evidence: `.test-artifacts/batch-mapping/complete_mapping.py` and
`complete-mapping.json`. Timing variation is preserved in the raw report.

Validation additionally covers single-frame empty/subset/reordered dictionaries,
unknown keys, shape errors, confidence bounds, confidence parity for all authored
body/hand mappings, composite and pass-through confidence, and chunk sizes 1/3/8
with four concurrent readers of one plan. Position comparisons use the existing
strict microbenchmark tolerance (rtol 1e-12, atol 1e-10); presence and confidence
comparisons are exact. Full FreeMoCap reference parity remains integration work.
