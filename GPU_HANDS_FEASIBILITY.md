# GPU hybrid hands feasibility acceptance

Status: experimental evaluation; the CPU MediaPipe hybrid remains official.
Frozen baseline: Tracker 713a175ae21fa0fdeeeb1cb6f5955e0fd02b2005 and Core
72235c4d5032c0cb2dd757996eb36c0a9d67cdbd. This protocol precedes candidate testing.

## Mandatory contract checks

- Reuse Tracker's shared 3x hand stages, clipped integer inference rectangles,
  One Euro crop filtering, parent keypoint names and prepared output assembly.
  The experiment replaces hand inference, not body inference or anatomical mapping.
- Preserve authored names/order, side identity and dense xyz/visibility arrays.
  Missing hand coordinates are NaN with zero visibility. No RTMW substitution.
- Verify image/crop/model coordinate transforms with analytical translated,
  rotated, non-square, clipped and empty examples. Coordinate round-trip error
  must stay below 1e-4 image pixels in these controlled float64 fixtures.
  This tolerance is a numerical transform check, not a hand-accuracy claim.
- Reject unsupported artifact signatures and non-finite configuration at setup.
  Any model-specific dimensions must derive from validated model/schema metadata.
- Test missing crops and partial batches without invented detections or camera
  reordering. Returned arrays must remain bitwise unchanged after subsequent
  frames, calls, other cameras, reset and shutdown.
- Preserve source timestamp handling. Existing CPU VIDEO state stays per camera.
  Explicitly document any experimental route without equivalent temporal ROI
  tracking; do not describe a stateless model as the complete MediaPipe task.
- Compare detector standalone and combined preset behavior. No Core, Forge or
  Cam imports in Tracker implementation/tests. No global session/sidecar edits
  without coordination with the other owners.

## Paired reference measurements

Use both complete prepared reference recordings (test and sample), verify video
hashes, pair identical source timestamps/images, alternate execution order, and
record model artifact hashes, providers, precision, thresholds and crop settings.
Use identical saved parent RTMW outputs/crops for an isolated hand-stage comparison,
then execute both complete hybrid pipelines for whole-tracking cost. Reusing
parent outputs is a measurement control, not a production data path.

Record per camera and hand: availability, missing-interval durations in seconds,
reacquisition delays after controlled visibility interruptions, coordinate
agreement on jointly available samples, and temporal jitter proxies. Retain dense
NPZ evidence for inspection. Availability is not correctness; reduced variation
can also mean lag. Neither proxy substitutes for annotated ground truth or live
human assessment. Report baseline repeatability and uncertainty, not an arbitrary
allowable loss of hands. Any observed quality regression remains an open tradeoff,
not an automatic pass based on faster timing.

Record cold preparation separately from warm runs. Measure hand-stage and complete
tracking latency (median/p95), including preprocessing, CPU/GPU transfers and
assembly. Exclude video decoding from tracking timings and report that scope.
Confirm actual provider execution; a requested CUDA provider alone is insufficient.
An acceleration recommendation requires improved complete-tracking median and p95
on both recordings across paired repeats, not just a faster network kernel.
Report the 30-Hz budget (33.33 ms per synchronized frame) as an application target,
not as a promise derived from the hand model's isolated inference time.

## Live assessment and lifecycle

GPU benchmarks and physical capture require the coordinator's exclusive slot.
Do not start cameras on this CPU preparation release. Reserve a later live visual
comparison with the user: open/closed hands, rotations, occlusion/reappearance and
camera transitions. The user's quality assessment is required before proposing
production selection. Measure stop/reset/close and retained outputs; test repeated
construction/close without leaked resources or later mutation of earlier results.

No candidate is promoted and no CPU fallback policy changes in this feasibility
phase. Step 6 proposes the separate production selection/fallback checklist from
the measured evidence, including unresolved quality or latency failures.

## Candidate audit (step 2)

The experimental candidate is the unquantized OpenCV Zoo February 2023 hand
model plus its companion palm detector, pinned to upstream commit
48b10fa505aeb51fbd127fc60a5c660040e823b7. The hand stage requires palm-aligned
preprocessing; it is not a direct replacement that accepts an arbitrary resized
hand rectangle. The source adapter rotates around the palm orientation, crops
and pads, converts BGR to RGB, and normalizes by 255. The hand network reports
image-space xyz, presence, handedness and a separate metric coordinate output.
Only the image-space keypoints belong in the current Tracker observation contract.
The preset's explicit side assignment stays authoritative. [@opencvZooHand2023]

| Artifact | SHA-256 | Verified raw input | Verified raw outputs |
| --- | --- | --- | --- |
| handpose_estimation_mediapipe_2023feb.onnx | db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c | input_1, float32 [1,224,224,3] NHWC | Identity [1,63], Identity_1 [1,1], Identity_2 [1,1], Identity_3 [1,63] |
| palm_detection_mediapipe_2023feb.onnx | 78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c | input_1, float32 [1,192,192,3] NHWC | Identity [1,2016,18], Identity_1 [1,2016,1] |

Artifact downloads matched the upstream Git LFS SHA-256 and size declarations.
ONNX checker passed; both raw models ran on CPUExecutionProvider with finite
outputs for zero-valued input. This verifies loading, not recognition quality.
No GPU execution has occurred. Unquantized artifacts avoid the upstream warning
about accuracy loss in its int8 variant. The model directories declare Apache-2.0;
that is separate from this repository's license. [@opencvZooHand2023]
The palm decoder uses SSD anchor centers, sigmoid scores and seven palm keypoints
for alignment. Its fixed-batch reshape constants need deliberate model preparation
before multicamera batching can be claimed. [@opencvZooPalm2023]

The official CPU route loads hand_landmarker.task with SHA-256
fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1;
its embedded hand detector hash is
945f713bc23570bd4ed60f848c401dc8eaf95713183d43ba14cf12e467d27a7d,
and hand-keypoint network hash is
6acda74af3fbf40e68265c20c7394b2bad81a16a481dcd79ad7a081887c3d6b9.
The public ONNX release and this task belong to the same model family, but
weight/conversion equivalence is NOT established by the upstream provenance.
The task includes temporal ROI tracking and conditional palm detection in VIDEO
mode. A stateless two-network adapter cannot claim equivalent tracking behavior
or equivalent speed. Any such experimental comparison must label this difference
and leave production selection unresolved until paired/live assessment.
[@mediaPipeHandTaskTracking]

Current session warmup accepts NCHW float32, while these raw graphs accept NHWC.
Use detector-owned, validated graph preparation and numerical equivalence tests
for layout/batch adaptation; do not silently reinterpret the raw graph's input.
The baseline registry collector described one model per detector; this hand task
needs two. The approved extension now collects plural model descriptors through
the factory while preserving shared-session ownership.

Pinned source, license, LFS pointers, graph inspection and CPU evidence are saved
in ignored .test-artifacts/w4/upstream/. Neither a source-code inspection nor a
successful zero-input inference is evidence that the hands track well.


## Experimental preparation checks

The pinned palm graph has fixed batch-one values in shared Reshape initializers
and Resize sizes. Candidate preparation makes those dimensions dynamic and wraps
the NHWC graph with the NCHW input expected by the existing session machinery.
CPU tests compare all raw and prepared outputs at batches 1, 3, 2 and 1.
The private palm-alignment adapter reuses Tracker image normalization and transform
helpers; its aligned image matches the pinned upstream implementation
[@opencvZooHand2023]. Exact inverse transforms preserve actual integer padding
positions, including odd padding, instead of approximating their centers.
These are preprocessing/numerical checks, not hand quality or GPU performance
results. Shared outer 3x crops remain unchanged. The detector registers only when its experimental module is imported; it is
not connected to the production preset.


## Step 3 integration status

The isolated factory now collects multiple descriptors per detector and the stage
accepts a dependent batch hook, followed by its existing coordinate translation,
filtering and assembly. Existing single-model processing stays the default.
Tests cover shared descriptor identity/collisions, camera IDs, source timestamps,
empty crops, reset/close ownership and independent outputs. Experimental recipe
construction preserves the production recipe's body, 3x crops and assembly.

The approved narrow sidecar exception now accepts coordinate-output stretch with
explicit coordinate scale after detector-owned alignment. Other estimator/resize
checks remain, and generic SimCC decoding still rejects coordinate outputs.
The experimental detector constructs and runs through the native factory on CPU.
Fresh validation: 69 native/sidecar tests and two reference construction tests passed.
The latter cover the first three synchronized frames from all three cameras in each
recording, not full-recording quality or GPU performance. The production preset is
unchanged. Step 4 evaluates paired recorded crops before GPU/live comparison.


## Step 4 recorded CPU evidence

Both full recordings passed paired, identical-crop comparisons: 222 test and
1,108 sample synchronized frames, three cameras each. Input video hashes remained
unchanged. Full-hybrid smoke checks also preserved non-hand RTMW outputs and
hand-crop rectangles exactly. Those full-hybrid checks cover three frames per
recording; the complete recordings compare the hand stages on baseline crops.

| Recording | CPU MediaPipe availability | Candidate availability | Median / p95 xy disagreement |
| --- | ---: | ---: | ---: |
| Test | 69.14% | 65.62% | 1.90 / 14.66 px |
| Sample | 75.11% | 66.26% | 1.69 / 11.52 px |

Availability counts finite keypoints with positive visibility. Disagreement uses
jointly available image coordinates, not ground-truth accuracy. Complete missing
intervals numbered 118/97 (CPU/candidate) for test and 200/303 for sample; longest
intervals were 3.00/4.67 s and 2.47/4.03 s. Recording-edge intervals are censored.
The candidate remains experimental; these results do not justify promotion.

Eight recorded-crop lifecycle cases passed independent-camera/batch equivalence,
partial/reordered batches, empty/blank handling, recovery for initially visible
crops, reset repeatability, and retained-result ownership. One sample left-hand
crop visible with preceding MediaPipe VIDEO history was not detected cold or
within the one-second recovery window. The reports explicitly preserve that
limitation. All selected candidate crops recovered on their first returning frame.

Evidence: Core `.test-artifacts/w4/paired-full.xml`, `body-preservation.xml`,
`cpu-paired-provenance.json`, `paired-crops/`; Tracker
`.test-artifacts/w4/recorded-lifecycle-verified.xml` (8 passed, 25.93 s).

Step 5 preflight also exposed a native MediaPipe Windows invalid-handle failure
(0xc0000008) after repeated tracker construction in one benchmark process. The
benchmark now isolates each repetition in its own process; this does not claim
to fix that native lifecycle failure. Preserve the initial Core
`.test-artifacts/w4/benchmark-smoke.log` for investigation. GPU measurements and
live quality checks remain pending; no production preset changes are proposed.

## Step 5 CUDA attempt and current blocker

The first granted CUDA comparison completed the baseline on the 222-frame test
recording, then the candidate process crashed during its first hand-model warmup
with Windows exception `0xc0000008`. No candidate frame, sample-recording run, or
CUDA node profile completed. The baseline's provisional tracking median was
69.42 ms and p95 was 104.63 ms after excluding the first 30 frames. Background
GPU activity was already present; this is not an idle-device comparison or a
candidate speed result. All seven owned process identities were verified gone,
device memory returned to its observed baseline, and the GPU slot was released.

CPU/read-only investigation found that the relevant W4/W5 CUDA-provider and
cuDNN DLL hashes and their loader function match. W4's crash report includes the
cuDNN sublibrary missing from W5's earlier failed attempt, so that earlier cause
must not be assumed here. Invocation/capture, session count, graph, warmup input,
threads, TF32, and arena settings still differ. The native cause is unresolved.
One-model CPU warmup probes passed for prepared batch3, prepared batch1, and the
original NHWC batch1 graph. A fresh bounded GPU assignment is required to test
the prepared standalone reproduction. Details and retained crash evidence:
Core `.test-artifacts/w4/gpu-comparison/diagnosis.md`. Step 5 remains open.

Application-readiness inspection also caught an unregistered experimental preset
name. The experimental recipe now retains the shared hybrid schema/mapping
profile while its detector configuration identifies the ONNX implementation.
The regression failed before correction; seven recipe/preparation checks and
both three-frame CPU reference smoke cases passed afterward. Full-recording
quality results above remain the earlier run. The narrow Core/Forge provenance
alignment is now applied and verified: the realtime manager imports, 27 focused
Core checks passed across the main run and a sandbox hard-link rerun, and two
additional checks consumed both reference recordings through current Forge
tensors without changing their Parquet files. The candidate still needs an
explicit application launch path before a live comparison. Neither the CUDA
blocker nor live readiness is resolved by these integration corrections.
# Current acceptance status — 2026-11-02

The coordinated live evaluation did not meet hand-quality or performance
acceptance. This detector remains an explicit experiment; it does not replace
the established RTMW plus MediaPipe hands route. CPU contract and lifecycle
checks below establish implementation behavior, not live tracking quality or
a GPU speed improvement. Earlier pending-live notes below are historical.
