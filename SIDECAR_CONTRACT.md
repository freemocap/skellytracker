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

`core/tracker/tracker_factory.py` adapts registry discovery from PR 90
[@skellytrackerFactoryIntegration]. `collect_required_sessions(config)` describes
the nested stage requirements without loading model files. `tracker_session_requests`
returns requests for `SharedSessionPool`; `build_tracker` and
`build_multi_person_tracker` instead allocate owned sessions and release them on
construction failure. Callers register detector classes by importing their modules,
as with the existing registries. Unsupported backends, missing model descriptors,
and conflicting descriptors with the same name fail before session allocation.
The factory is available directly from its module; Core's current factory is
unchanged until the separate Global 2 migration.

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
