# RTMW with MediaPipe hands

## Ownership and contract

Tracker owns the preset, detector composition, keypoint output assembly, schema,
connections, and mapping selection. Core supplies user settings/images, calls
Tracker, and connects its outputs to recording and Forge. Forge owns anatomical
landmark definitions and reconstruction. No sibling imports are required here.

The preset selects RTMW body/feet/face keypoints and MediaPipe hand keypoints.
RTMW still computes its wholebody model; replacing its hand results does not
remove that model's inference cost. MediaPipe retains its actual sided keypoint
names, including `right_hand_index_finger_mcp`; results are not relabeled as RTMW
`right_hand_forefinger1`. The existing independent RTMW preset is unchanged.

The output order is RTMW's authored order excluding its sided hand keypoints,
then right and left MediaPipe hands in their authored order. Counts and numeric
selection indices derive from the schemas, not hard-coded offsets. Core can sort
the published channel order using its existing channel contract. The current
Keypoints API has xyz and visibility arrays; this work does not redefine that API.

Both presets and the standalone cropped-hand demo use mediapipe_hand_stage.
The crop center averages the parent's index/pinky tips; square size is three times
the larger valid wrist-to-tip distance, with a 30-pixel minimum. The preset
selects only the parent keypoint names: MediaPipe index/pinky or RTMW
hand_forefinger4/hand_pinky_finger4. RTMW finger estimates guide cropping but never
replace missing MediaPipe output. If both guiding tips are unavailable, the crop
is skipped; this remains a limitation to assess during live use. Each cropped
MediaPipe detector requests one hand with an explicitly assigned side.
Body and hand crops use the same dense-array One Euro arithmetic as keypoint
filtering. Each camera/stage retains its own corner workspace. Defaults are
3 Hz minimum cutoff, beta 0.02 and 1 Hz derivative cutoff. There is no added
percentage expansion. Missing crops clear filter history; reacquisition starts
at the detected crop. A source-time gap above 0.5 seconds also resets history.
The transmitted crop is the exact clipped integer pixel rectangle used for inference.

Pass source `timestamp_ms` to both single-image and batch processing. Recorded
video processing derives it from recording time; live callers use capture time.
Missing, duplicate or backwards timestamps bypass/reset smoothing rather than
assuming a frame rate. Preserved observation arrays and boxes remain independent
of later state updates.

An absent hand produces NaN coordinates and zero visibility for that hand's
declared keypoints. Never substitute RTMW hand estimates silently. Body wrists
remain distinct from MediaPipe hand wrists. The preset selects the existing
RTMW body mapping and MediaPipe hand mapping; those mappings retain their current
choice of body wrist for the anatomical wrist landmark. Reconstruction/filtering
policy is unchanged.

Assembly uses prepared NumPy indices and array copies, not per-keypoint value
dictionaries. Returned observations own their values: subsequent frames/cameras
must not overwrite retained outputs. Both single-image and batch callers receive
the same assembled schema. Unrelated stages such as Charuco are preserved.

## Verification

Check schema identity/order, declared connections, mapping composition, missing
hands, schema mismatch errors, independent observation storage, and session
requirements without native inference. Then exercise real models on both reference
recordings through Core after its integration stage. Report processing time and
hand availability separately from human assessment of tracking quality.

## Tracker implementation and integration handoff

`core/tracker/tracker_presets.py` provides `combined_tracker_config` and the
existing RTMW/MediaPipe recipes for consumer migration. `core/io/tracker_profiles.py`
provides names, connections, and mapping selection. `TrackerConfig.keypoint_assembly`
describes source schemas and selected keypoints; Tracker prepares indices once and
applies the same assembly in single-image and batch processing. The MediaPipe-only
recipe also declares its output assembly. The old merge helper is available from
Tracker for consumer migration; the combined preset does not use concatenation.

Core now delegates recipe/schema/mapping ownership to these interfaces and exposes
the combined option throughout its API/UI/pipelines. Both live implementations and
both post-hoc implementations passed both complete reference recordings. Core owns
these cross-project tests; Tracker tests do not import Core or another subskelly.

Validation on 2026-10-31: 56 tests passed (52 focused contracts/regressions and four
real-model checks). Both reference recordings passed single-image and three-camera
batch processing, each using their first 12 frames per camera. Total: 144 camera
observations. Finite hand-keypoint counts were 231 for test data and 714 for sample
data in each mode; these are counts over the excerpts, not accuracy measurements.
CPU ONNX and MediaPipe ran in Tracker's existing environment. These short checks
do not establish quality improvements, full-recording acceptance, or live UI speed.
Full Core reference-data comparison remains step 4.5 after consumer integration.

Focused command (from Tracker):

```powershell
.venv/Scripts/python.exe -B -m pytest skellytracker/tests/test_tracker_presets.py skellytracker/tests/test_keypoints.py skellytracker/tests/test_tracker_factory.py skellytracker/tests/test_combined_tracker_reference.py -q -p no:cacheprovider -o junit_family=xunit1 --junitxml=.test-artifacts/preset-final.xml
```

Reports are ignored under `.test-artifacts/`. Static undefined-name/import checks
and `git diff --check` passed. The initial Tracker implementation is commit
`0a59be6`; subsequent integration helpers and validation remain working-tree changes.

## Custom ONNX validation (step 4.4)

`tests/test_custom_onnx_loading.py` creates a synthetic local ONNX artifact and YAML
sidecar, resolves it through `sidecar_model_spec` into `OnnxModelSpec`, and executes
it with the real CPU `OnnxSession` at batch sizes one and three. Output values match
the independently computed NumPy result, network access is forbidden, and sessions
are closed. The related sidecar contract suite passes 23 tests. These synthetic
fixtures test loading and execution, not a trained DLC model or tracking accuracy.

The current limits are explicit:

- The schema accepts heatmap and coordinate descriptions, but the implemented
  pose decoder supports two-dimensional SimCC. Both unsupported methods are tested
  to raise rather than being interpreted as SimCC.
- Session warm-up currently constructs float32, three-channel NCHW input.
  An NHWC descriptor can resolve its height and width correctly but still fails
  session creation; a real synthetic NHWC artifact tests that failure. Other input
  layouts/dtypes need a matching warm-up contract before they can be claimed as
  supported. FP16 weights with float32 input are a different case.
- A sidecar does not instantiate a generic detector or check all model output
  metadata against the artifact. The detector adapter must validate its accepted
  input/output signature and implement preprocessing, decoding and coordinate
  restoration.

The existing `rearchitecture-docs/skellytracker-architecture/dlc-independent-inference.md`
describes the proposed independent DLC importer. For its first supported model,
the remaining concrete work is an accepted architecture/weights importer or an
already exported artifact; exact image transforms and tensor metadata; ordered
keypoint names and connections; and a decoder that honors that model's heatmap,
location-refinement, stride and score settings. Validate those against known model
outputs before composing the detector through Tracker's registry. No DLC package,
importer, trained artifact or decoder was added by step 4.4. Anatomical mapping and
reconstruction compatibility remain separate from successful image inference.

## Complete-recording comparison (step 4.5)

Core's `freemocap.tools.datasets.hand_tracker_comparison` ran both full reference
recordings twice per preset, reversing order on the second repeat. With CUDA RTMW
on an RTX 3090 and CPU MediaPipe hands, mean three-camera tracking-call times were
14.19 versus 55.37 ms on test data and 12.65 versus 47.87 ms on sample data.
Available hand-keypoint fractions were 95.45% versus 21.70%, and 95.74% versus
26.50%, respectively. These counts are not accuracy measurements. The combined
recipe still runs RTMW whole-body inference before the additional hand passes.

All 91 retained RTMW keypoints and visibility values matched exactly throughout
every repeat. Core's `TRACKER_COMPOSITION_INTEGRATION.md` records methods, limits
and the live comparison procedure. The combined preset is opt-in; its crop and
threshold settings are not optimized defaults. Final Tracker validation passed
76 tests, including a fresh-process check that preset registration does not need
Core, SkellyCam or SkellyForge imports.

## GPU feasibility and crop visibility — 2026-11-01

Core exposes `RTMW + MediaPipe hands` separately from the full `MediaPipe` preset.
The latter passes the full image to the body task at Tracker's boundary; that
recipe has no external person detector, as was also true in Core before migration.
MediaPipe performs its own internal detection/tracking. The hand child stages
still use explicit Tracker crops. At the initial audit, output assembly removed consumed child
observations after copying their keypoints, losing crop boxes. The crop correction
below now preserves those child stages as metadata without duplicate keypoints. Missing overlay boxes do not establish missing cropping.
The previous Core merge helper also removed those children.

The initial audit found that hand, face and body task constructors omitted an
explicit delegate. Installed `BaseOptions.to_ctypes` maps that omission to CPU.
The session's `_probe_gpu` also omitted the delegate, so its success log falsely
claimed GPU mode without testing GPU inference. The subsequent approved correction
removes that probe and explicitly requests CPU in all three task constructors.
The session now reports CPU; an explicit `disable_gpu=False` request raises a clear
unsupported-execution error. No camera pipeline or application server was restarted.
The upstream Python options also document platform limits for GPU support.
[@mediaPipePythonDelegateOptions]

An ONNX route is feasible: OpenCV Zoo publishes converted MediaPipe hand and body
models. These are candidates, not proof of identical weights or quality to our
current `.task` files. Start with floating-point artifacts and numerical comparison
to the current CPU task; the hand model documentation warns about quantization
accuracy loss. [@opencvZooMediaPipeHandOnnx; @opencvZooMediaPipePoseOnnx]

The hand adapter requires more than a filename: its reference path uses a rotated
palm crop, NHWC input and inverse coordinate transforms, plus separate presence,
handedness and coordinate outputs. Our current warm-up assumes NCHW, and the
existing pose decoder handles SimCC. A Tracker-owned adapter must supply these
missing contracts. A wrist-centered crop alone is not proof of equivalence to
the model's required oriented palm crop. [@opencvZooMediaPipeHandTransforms]

The sidecar, registry, session and dense assembly infrastructure can be reused.
Validate supported batch dimensions before batching hand crops across cameras;
validate per-camera/per-hand tracking state and reacquisition behavior as well as
single-image predictions. MediaPipe's task includes tracking behavior beyond the
neural model. [@mediaPipeHandTaskTracking]

Moving inference to ONNX CUDA is a candidate optimization, not a measured speedup
or a guarantee of 30 Hz. Crop preparation, inference, temporal tracking and
transfers need separate profiling. The current whole-body RTMW pass still precedes
the hand pass; GPU execution does not remove that dependency. This inspection
made no runtime changes; the later device-reporting correction is described above.
Neither step downloaded models, installed dependencies or accessed cameras.

Device-reporting validation: 21 tests passed, including actual task option checks
for hand/body/face in image/video modes, import order, session lifetime, and native
hand output equality to the prior implicit CPU default on both reference datasets
(three frames per camera). A shared-session mock was updated for the assembly
field added earlier. Core's `TRACKER_COMPOSITION_INTEGRATION.md` records remaining
crop-publication, hand-quality and GPU-adapter work; none is marked complete.


## Hand crop correction — 2026-11-01

The hierarchy was broken for nonzero parent crops: it passed a person-cropped
image to hand children but supplied wrist coordinates in the source image.
A deterministic pixel-coded-image regression reproduced an intended 80x80 crop
being delivered as a misplaced 20x40 crop in single, batch and associated-person
processing. Children now receive the same source image as their parent keypoints;
only their own crop is sliced. No image copy is added.

Bounding boxes published in StageObservation describe the actual clipped integer
pixel slice, including smoothing, rather than the raw detector proposal. Keypoint
translation uses that same integer origin. A missing required crop suppresses
inference instead of falling back to an unrelated full-image hand search.

Prepared assembly and the merge helper retain child crop metadata while clearing
only child keypoint references after copying values to the assembled tensor.
TrackerProfile declares crop_stage_paths. RTMW declares its body crop; the hybrid
also declares both hand crops; pure MediaPipe declares only its hand crops, not
full-image body/face fallback rectangles. Core consumes this declaration through
its existing BOXES_2D channel and renderer.

Validation: 38 focused hierarchy/preset/factory/multi-person tests passed (including
integer and fractional parent origins), plus four real-model reference tests.
The reference tests exercised 12 frames per camera, three cameras, both recordings,
and both single and batch paths: 144 observations total. Single/batch counts agreed
at 714 finite hand keypoints for test data and 987 for sample data per 36-observation
run. These counts are availability, not accuracy or a matched quality improvement.
Hand crop bounds were checked against source image dimensions in the batch runs.
No live cameras, model weights, thresholds or dependencies were changed.


## Shared hand crop policy - 2026-11-01

The hybrid previously built the shared hand stage and then replaced its crop
configuration. That was setup-time replacement, not two calculations per frame.
The override is removed. Parent-schema names are selected inside the shared
factory before constructing the config. The existing KeypointBoundingBoxDetector
performs the crop calculation. The standalone MediaPipe demo also uses this
factory instead of a separate copy. No Core production code or dependency changed.

Sixteen geometry cases first failed against the old rules. They now pass for both
sides, both presets, and four orientations, including a foreshortened forearm,
partial/missing fingertips, and wrist/fingertip coverage. Demo parity and invalid
schema tests also pass. Validation: 52 focused Tracker tests across runs, eight
native-model reference checks, and 15 Core integration tests. Native checks cover
both presets and both recordings in single and batch modes: 12 frames per camera,
three cameras per recording; 288 camera observations overall.

Matched hybrid checks in the same environment produced these finite hand keypoint
counts per 36-observation excerpt; single and batch results agree:

| Recording | Previous wrist/forearm crop | Shared fingertip crop, 4x size |
| --- | ---: | ---: |
| Test | 714 | 840 |
| Sample | 987 | 1323 |

These are availability counts, not accuracy or throughput measurements. Reports
are ignored under .test-artifacts/crop-policy-{before,after,mediapipe}.xml. Live
visual acceptance remains open. No cameras or running application were touched.
