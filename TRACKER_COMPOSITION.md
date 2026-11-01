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

Each hand crop is centered on the corresponding RTMW body wrist. Its square size
uses the same-side elbow-to-wrist distance, initially scaled by 1.5 with the
existing 30-pixel minimum. Thus missing RTMW finger detections do not prevent
MediaPipe from finding a hand. Invalid wrists suppress that hand crop. These are
initial evaluation settings, not empirically optimized settings. Each cropped
MediaPipe detector requests one hand with an explicitly assigned side.

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
