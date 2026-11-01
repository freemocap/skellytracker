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

Core migration is pending: replace its recipe/schema/mapping implementations with
these interfaces and expose the combined option throughout its API/UI/pipelines.
No Core runtime changes have been made in the Tracker implementation stage.

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
and `git diff --check` passed. Changes are unstaged for the human-owned Git handoff.
