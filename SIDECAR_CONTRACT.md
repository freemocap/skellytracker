# Supported sidecar contract on sfn26

PR 84 is integrated with its complete branch history [@skellytrackerSidecarImplementation],
including standalone detector migrations, RTMW module relocation, overlay groups,
mapping expansion, schemas, specifications, and tests. Compatibility fixes retain
sfn26's confidence, landmark ordering, model-download and session-ownership contracts.
PR 81's branch history, resize additions and proposed exporter specification are
integrated [@skellytrackerSidecarExtensions]. The specification's `vNEXT` sections
describe future functionality; they do not declare a released schema version.
Accepted fields are defined by `core/sidecar/model.py`; unknown fields fail validation.

UTF-8 YAML composition is resolved at initialization. `sidecar_model_spec` creates
descriptors without network/file I/O; normal session creation resolves artifacts.
Fresh remote downloads verify declared SHA-256 before atomic publication inside
existing file locks. ZIP hashes cover downloaded archives. Already cached files
retain the existing cache trust policy; local paths ignore URL hashes.

RTMW wholebody and YOLOX consume the sidecars. RTMW native ordering is mapped to
the established public ordering through an immutable precomputed index array.
Confidence remains the mean of raw SIMCC peaks, clipped to [0, 1], with the 0.4
threshold. Declared landmark counts and split-ratio/bin counts are checked.
Normalization functions and model sizes are prepared before frame processing.

## Implemented preprocessing

- Nonempty uint8 BGR images; float32 model inputs; NCHW or NHWC output layout.
- Existing letterbox declarations retain top-left placement, integer-truncated
  dimensions and their scalar-ratio inverse. Explicit padding/rounding settings
  support top-left or centered placement and half-even or half-up rounding, with
  inverse coordinates derived from the actual resized dimensions and offsets.
- Stretch preserves independent x/y scales. Full-frame affine fit preserves its
  uniform scale and fractional offsets. Both return `ImageTransform` metadata;
  generic box decoders apply its vectorized inverse without modifying received arrays.
- Letterbox and fit accept scalar or per-channel BGR padding before normalization.
- Affine person crops with configurable expansion and interpolation, target aspect
  ratio preservation, zero padding, and center/scale inverse metadata.
- None, unit-float, ImageNet BGR/RGB and custom normalization; RGB modes reorder
  incoming BGR channels before normalization.

## Unsupported functionality

The proposed `decode.selection`, output-binding extensions, `input.color_format`
and other unimplemented `vNEXT` fields are rejected by the strict parser. Resize `none`,
non-aspect-preserving affine cropping, heatmap/coordinate pose decoding, 3D SIMCC
decoding and non-float32 preprocessing raise explicit errors. Declaring artifact
variants does not export them: no automatic static-batch converter, hardware-aware
precision selector or generic sidecar-only detector implementation is included.

Standalone RTMPose body, hand and face now use the sidecar pipeline. The body models
actually emit 17 points, which now agree with their schema and dedicated body mapping;
the 23-point body mapping used by wholebody composition remains separate. All these
model variants were exercised with verified weights and compared against the original
preprocessing and decoding on the same CPU session. RTMW's implementation now lives
under `rtmw/wholebody`; the old Python imports re-export the same classes. The public
wholebody schema path remains available for current Core consumers until Global 2.

## Verification

`test_sidecar_compatibility.py` checks public landmark ordering, raw confidence and
affine numerical parity. `test_sidecar_supported_contract.py` checks channel order,
interpolation and rejection of unsupported declarations. Upstream foundation tests
are retained. YOLOX real-model tests compare existing and new processing on the
same backend; upstream golden files do not identify their backend/precision.
Optional model variants without cached weights are reported as skipped.

## Tracker construction

`core/tracker/tracker_factory.py` integrates registry discovery from PR 90
[@skellytrackerFactoryIntegration]. `collect_required_sessions(config)` describes
the nested stage requirements without loading model files. `tracker_session_requests`
returns requests for `SharedSessionPool`; `build_tracker` and
`build_multi_person_tracker` instead allocate owned sessions and release them on
construction failure. Callers register detector classes by importing their modules,
as with the existing registries. Unsupported backends, missing model descriptors,
and conflicting descriptors with the same name fail before session allocation.
The factory builders, `collect_required_sessions`, and `tracker_session_requests`
are exported from `skellytracker.core.tracker` as well as their defining module.
The standalone builders own their sessions; shared callers acquire tracker leases
from `SharedSessionPool` using the same factory-generated requests. Closing a lease
releases its detector state, while model sessions remain alive until the last lease
closes. Batch-size and provider overrides participate in session matching.
Core's current factory is unchanged until the separate Global 2 migration.

## Initial selective-port validation — 2026-10-31

The focused suite passed 241 tests; one optional YOLOX-tiny model test skipped
because its weights were not cached. Eight bibliography-validator tests passed;
43 references validated without errors. Existing batched detector correctness
tests now select CPU explicitly instead of depending on installed provider order.
The reference-frame comparison used CUDA explicitly.

Test and sample recordings each supplied 12 frames, tested with one camera and
three cameras: 96 camera-frame observations in total. Landmark ordering and NaN
masks matched, and maximum absolute coordinate and confidence differences were
both zero relative to Tracker b535c0f. These are short inference checks, not
whole-recording processing or camera-to-display latency benchmarks.

| Reference case | Baseline median (ms) | Integrated median (ms) |
| --- | ---: | ---: |
| Test, one camera | 21.817 | 21.209 |
| Test, three cameras | 39.319 | 38.748 |
| Sample, one camera | 20.401 | 20.466 |
| Sample, three cameras | 37.600 | 38.801 |

The small differences do not establish a speedup or regression. Each median
excludes two initial frames. Final isolated reports and array comparisons are
under the outer workspace's ignored `.test-artifacts/sidecar-integration/`,
in `baseline/` and `handoff-factory/`. An earlier `verified-factory/` run overlapped
the CPU test suite and must not be used for timing comparison.

Tests cover actual mixed CPU/MediaPipe/ONNX session construction, shared-session
reuse, and cleanup after partial detector, stage, or session construction failure.
They do not validate the future RTMW-body/MediaPipe-hand composition or its speed.
Synthetic decoder tests also cover raw YOLOX anchors without mutating input,
rejection of unsplit NMS camera batches, and NHWC spatial dimensions.

## Full PR 84 merge validation — 2026-10-31

The combined affected suite passed 376 tests without skips. Eight bibliography
validator tests passed, with 43 references and no citation errors. The CUDA reference
check again produced zero coordinate and confidence differences for all 96
camera-frame observations; its report is in the outer workspace's ignored
`.test-artifacts/sidecar-integration/merge84-reference/`. This is correctness evidence,
not a full-pipeline performance measurement.

Additional model weights were downloaded with declared SHA-256 verification into
Tracker's ignored `.test-artifacts/sidecar-integration/merge84-models/`. Set
`SKELLYTRACKER_TEST_MODEL_CACHE` to that directory and `SKELLYTRACKER_TEST_PROVIDERS=cpu`
to repeat the expanded tests. Tests cover both body sizes, hand, face, all three RTMW
sizes, and both YOLOX sizes. Upstream golden JSON is retained as historical evidence;
because it lacks provider/precision provenance and includes the previous confidence
contract, regression checks compare the original math with the new path on the same
session, including confidence, missingness, and coordinate values.

## Full PR 81 merge validation — 2026-10-31

The combined affected suite passed 388 tests without skips. Resize-specific checks
cover half-even/half-up ties, centered odd remainders, scalar and per-channel padding,
fractional fit offsets, stretch scale differences, inverse mapping through both
generic detection decoders, input-array preservation, and existing letterbox behavior.
The CUDA reference comparison is recorded under the outer workspace's ignored
`.test-artifacts/sidecar-integration/merge81-reference/`.

The merged specification and exporter coverage plan are retained in full. A
documented proposed exporter format is not an implemented decoder: new detection
selection modes, output bindings, and exporter adapters still require dedicated
implementation and model validation before their schema version can be released.

## Full PR 90 merge validation — 2026-10-31

The combined affected suite passed 405 tests without skips. The incoming nested
factory tests are retained in `test_tracker_factory_public.py`, using the public
requirement-discovery API. Existing shared-session, descriptor-conflict, batch
override, and partial-construction cleanup checks remain intact. All seven migrated
demo/example modules import successfully; no cameras or GUI were started.

`test_factory_shared_inference.py` creates two independent trackers from the same
factory requests, verifies a single real ONNX session allocation, closes one
tracker, and confirms the other still produces identical coordinates and confidence.
The models are released after the last lease closes. MediaPipe/CPU construction
and inference are also exercised twice in a separate process. Two initial combined
pytest runs raised Windows native exception `0xc0000008` in MediaPipe's C dispatcher.
The focused test passed in-process; isolating its native lifecycle also made the
combined suite pass. The native exception's underlying cause is not established;
isolation propagates subprocess failures without terminating unrelated tests.
No runtime workaround or dependency changes were introduced for this test issue.

The CUDA reference comparison again shows zero coordinate and confidence changes
for 96 observations across test/sample recordings and one/three-camera batches.
Reports are in the outer workspace's ignored
`.test-artifacts/sidecar-integration/merge90-reference/`; the combined pytest log is
in Tracker's ignored `.test-artifacts/sidecar-integration/merge90-validated.log`.
Eight bibliography-validator tests passed; 43 references have no citation errors.

## Combined integration validation, step 1.5 — 2026-10-31

Validation used the existing Tracker and Core environments and local editable
sources, without dependency installation or lockfile changes. Core runtime code
was not migrated in this step. Its reference tests now support both recordings,
and a new 48-case test compares Core's existing session requests with Tracker's
registry factory without allocating sessions.

The expanded checks found and fixed an import-order defect: importing
`MediaPipeSession` before the detector registry caused a circular import through
the MediaPipe model manager. That helper is now imported inside the GPU probe.
Four fresh-process tests verify session/factory imports in different orders,
without importing native MediaPipe or ONNX Runtime at module import time.

The native Windows exception recorded above was reproduced with pytest's `fd`
capture during repeated mixed MediaPipe/CPU construction. The same six-cycle
diagnostic passed with `sys` capture, which leaves native output handles alone.
Tracker now defaults to `--capture=sys`; three repeated in-process lifecycle
tests and the existing subprocess lifecycle test pass. This is a test-capture
adjustment, not a production logging or inference workaround. The data-store
video fixture also now supplies recording-derived timestamps and guarantees
tracker cleanup after errors; its previous processing-clock timestamps could
violate MediaPipe's monotonic timestamp requirement.

Verified results:

- Tracker non-video suite: 593 tests plus four subtests passed, with no skips.
- Tracker video suite: 46 checks passed on test data and 46 on sample data,
  including explicit CPU/CUDA cases, MediaPipe, ArUco, ChArUco and data-store
  consumers. These detector tests use bounded frames from each recording;
  Core's pipeline runs below process complete recordings and all three cameras.
- Core factory/inference/cancellation/replay-driver checks: 23 tests plus two
  subtests passed. Session-request parity: 48 cases passed. Dense tensor,
  memory-map ownership, source timing and replay consumers: nine tests passed.
- All 16 real-time replay cases passed across both complete recordings:
  centralized/per-camera inference, explicit/latest calibration selection, and
  board-only/board-plus-human tracking. Eight board cases passed in the initial
  matrix; eight human cases passed after allowing provider-cache access.
  Tests check frame identity, source timing, live filter updates, reconstruction,
  shutdown and shared-memory release, while preserving source-file hashes.
  Reference data can still produce unsolved hand segments and camera-geometry
  warnings handled by the configured continue policy. Passing reconstruction
  assertions do not mean every segment is solved on every frame.
- All 96 camera-frame comparisons against Tracker `b535c0f` passed again.
  Coordinates, confidence values and missingness masks were exactly equal;
  authored landmark ordering matched. These are the first 12 frames of each
  recording with one/three-camera batches, not whole-recording equivalence.
- Bibliography validation: 43 references, no errors; eight validator tests passed.
  Focused fatal-error lint and whitespace checks passed.

The default graph post-hoc pipeline completed fresh calibration, tracking,
triangulation, alignment, filtering, scale fitting, reconstruction and Parquet
validation for both full reference recordings:

| Recording | Frames per camera | Cameras | Parquet rows | Calibration error (px) | Worker lifecycle (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test | 222 | 3 | 1,865,466 | 0.375611 | 40.790 |
| Sample | 1,108 | 3 | 9,310,524 | 0.408939 | 153.451 |

The selectable older `current` post-hoc implementation also completed the test
recording with 222 frames and 1,865,466 rows. Its run overlapped other correctness
tests and is not timing evidence. Graph worker-lifecycle times include setup and
output work; they do not establish an inference speedup or live-camera frame rate.
All processing used the explicit `numerical` output profile. Blender, GUI and
hardware-camera acceptance were not run in this stage.

JUnit and diagnostic logs are ignored under each repository's
`.test-artifacts/sidecar-integration/stage15-*`. The final array comparison is in
the outer workspace's `stage15-reference/` directory under that artifact root;
its concurrent-run timings must not be used as performance comparisons.
Prepared graph recordings and `ready.json` reports are under
`%LOCALAPPDATA%/Temp/fc-sc15-20261031/`; the older implementation uses
`%LOCALAPPDATA%/Temp/fc-sc15-current-20261031/`. Original recordings were not
overwritten. Initial fixture-download and Windows sandbox cache/temp-directory
errors were resolved before the passing reruns; failed-attempt logs are retained.
