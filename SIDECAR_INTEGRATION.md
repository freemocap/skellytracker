# Sidecar integration into sfn26

Approved 2026-10-31; revised by the owner to use full Git merges.
Runtime integration is limited to SkellyTracker in Global 1.
The owner explicitly authorized the Git changes needed for these merges, with
short commit messages and a pause after each PR for the owner to post a comment.
Core migration follows the completed Tracker integration.

## Approved global checklist

- [x] **Global 1. Integrate the sidecar implementation and registry-based tracker factory into SkellyTracker `sfn26`, with verified compatibility and a reviewed Git handoff.**
- [x] **Global 2. Migrate FreeMoCap’s model loading and tracker construction to the new SkellyTracker interfaces, then validate live and post-hoc processing against reference recordings.**
- [ ] **Global 3. Adapt Core PR #931’s execution-provider controls to the current API, UI, and shared inference service, then verify selection and failure behavior.**
- [ ] **Global 4. Implement and evaluate RTMW-body/MediaPipe-hand composition, and validate the custom-ONNX loading interface needed by the DLC work.**
- [ ] **Global 5. Benchmark supported precision, batch-size, and execution-provider combinations against the established baseline, then recommend which acceleration changes to adopt.**

## Approved Global 1 checklist

- [x] **1.1. Commit the existing SkellyTracker fixes and tests, and verify the three PR revisions and their merge order.**
- [x] **1.2. Merge PR #84 into `sfn26`, resolve compatibility issues, validate and push the result, then pause with its PR comment.**
- [x] **1.3. Merge PR #81 into `sfn26`, reconcile its complete sidecar contract, validate and push the result, then pause with its PR comment.**
- [x] **1.4. Merge PR #90 into `sfn26`, reconcile its factory with shared-session ownership, validate and push the result, then pause with its PR comment.**
- [x] **1.5. Validate the combined SkellyTracker changes against reference recordings and all affected detector interfaces.**
- [x] **1.6. Verify the published SkellyTracker revision and prepare the FreeMoCap integration handoff.**

The previous selective-port implementation and its passing tests are preserved
in commit `d7b0b1a`. Its completed checks are evidence to preserve, not completion
of the requested full merges. `SIDECAR_CONTRACT.md` records each integration's
supported behavior and validation evidence.

The owner explicitly clarified that this task's Git authorization supersedes the
older workspace prohibition. Approval review accepted the resubmitted fetch.
PR histories are available locally; proceed with the authorized merges and
pause after each validated, published PR integration.

PR 84 merged and published as `9668a8d40881a09db083744db4b9b9f2639f7087`.
Its second parent is the exact PR head, and the published branch was verified.
Validation: 376 tests passed without skips, eight bibliography tests passed,
and all 96 reference camera-frame observations had zero coordinate/confidence
differences from the original baseline. The owner posted the PR 84 comment and
authorized proceeding to PR 81.

PR 81 merged and published as `4a8b1f7fd8175c28d6815a7c42fa8deffa9f10d5`.
Its second parent is the exact PR head, and the published branch was verified.
Validation: 388 tests passed without skips, eight bibliography tests passed,
and all 96 reference camera-frame observations had zero coordinate/confidence
differences. The full proposed vNEXT specification is retained, with implemented
resize behavior and unimplemented exporter features clearly distinguished.
The owner posted the PR 81 comment and authorized proceeding to PR 90.

PR 90 merged and published as `c876aba8b75cec550b363970a4c84656b9c63436`.
Its second parent is the exact PR head, and the published branch was verified.
Validation: 405 tests passed without skips, eight bibliography tests passed,
and all 96 reference camera-frame observations had zero coordinate/confidence
differences. The public factory and examples retain shared-session discovery,
strict descriptors, batch overrides, and construction-failure cleanup. Two trackers
shared real ONNX models successfully through independent close operations.
The new MediaPipe construction test runs in an isolated subprocess after native
Windows exceptions in combined runs; see `SIDECAR_CONTRACT.md` for the limitation.
The owner subsequently authorized steps 1.5 and 1.6, including publishing fixes
and tests. Step 1.5 is complete: 593 Tracker tests plus four subtests, 92 reference
video checks, all 16 Core real-time replay cases across both full recordings,
23 Core integration tests plus two subtests, 48 session-request parity cases,
and nine tensor/timing consumer tests passed. Default graph post-hoc processing
passed both full recordings; the older selectable pipeline also passed test data.
All 96 bounded camera-frame comparisons remain numerically identical to the
original baseline. See `SIDECAR_CONTRACT.md` for fixes, commands and limitations.
Step 1.6 is complete. Tracker fixes and validation were published as
[`148dcee`](https://github.com/freemocap/skellytracker/commit/148dcee82c30550e740f36facc9e7cd987237ca5),
and Core's reference-test expansion as
[`ccfda109`](https://github.com/freemocap/freemocap/commit/ccfda109f4c1d5c0c67539a2eb2c8069990b4a5e).
Both `origin/sfn26` SHAs were verified with `git ls-remote` after pushing.
All three exact PR heads remain ancestors of the published Tracker revision.
No generated artifacts were staged. Tracker was clean; Core's pre-existing
100-line `references.bib` addition was preserved unstaged. The following
documentation checkpoint recorded Global 1 completion; Global 2 completion is
recorded below.

## Historical handoff before Global 2

The following describes the starting state and approved migration boundaries.
Global 2 is now complete; see its published record below.

Core still runs its existing factory in
`freemocap/core/tracking/tracker_factory.py`. The passing integration runs test
that consumer against the merged Tracker; they do not claim Core has adopted
the new factory yet. No package versions or lockfiles changed in Global 1.

1. Replace Core's hard-coded RTMPose/YOLOX resource discovery with the public
   `skellytracker.core.tracker.tracker_session_requests` API. Map Core's
   `execution_provider` into `onnx_overrides` and `mediapipe_disable_gpu` into
   `mediapipe_overrides`; preserve `batch_size`. Core's new
   `test_sidecar_factory_contract.py` verifies exact request equality for 48
   combinations before migration.
2. Preserve allocation ownership. `freemocap/core/pipeline/inference_service.py`
   owns its `SharedSessionPool` and tracker leases on one managed thread. Feed it
   requests; do not replace pooled trackers with independently owned
   `build_tracker` instances. Preserve per-client state, cancellation, lease
   cleanup and session reuse. Direct construction in `build_configured_tracker`
   and `build_skeleton_onnx_session` must retain `_onnx_session_build_lock`;
   Tracker's standalone convenience factory does not supply that Core lock.
3. Update canonical RTMW imports and schema access deliberately.
   `freemocap/core/tracking/tracker_definitions.py` still reads
   `rtmpose/wholebody/rtmpose_wholebody.yaml`. The compatibility classes/YAML
   remain available; canonical implementation is `rtmw/wholebody/`.
   Preserve saved model/stage names, confidence semantics, authored landmark
   ordering and skeleton connections while changing package access. Core owns
   bounding-box defaults, model composition and observation adaptation.
4. Repeat the factory/inference tests, both full real-time reference replays and
   both graph post-hoc runs after migration. Compare numerical values and source
   timing, and verify worker/session cleanup. MediaPipe tests using pytest's
   native file-descriptor capture need the documented `sys` capture setting.

The proposed vNEXT exporter specification is not a universal custom-model
decoder. Unsupported output bindings/exporter adapters still need model-specific
validation in later stages. RTMW-body/MediaPipe-hand composition, DLC integration,
provider controls and acceleration benchmarks remain Globals 3–5. Existing
automatic-provider runs used CUDA; missing TensorRT runtime was not installed.

### Proposed Global 2 checklist — awaiting approval

- [x] **2.1. Replace Core's hard-coded session discovery with Tracker's registry requests, preserving shared-session ownership and ONNX construction locking.**
- [x] **2.2. Update Core's RTMW imports and schema access, with tests for model names, confidence values, landmark ordering and skeleton connections.**
- [x] **2.3. Validate the migrated Core factory through real-time replay and post-hoc processing of both reference recordings.**
- [x] **2.4. Publish the tested Core integration and prepare the execution-provider controls handoff.**

Global 2 was approved and completed on 2026-10-31. Tracker's public RTMW schema
accessor is published in `4118b5481458e2383719fc6535d559dc8f2a51b9`. Core's runtime
migration is published in `38c67163c7fb0b0f15647edd92ab1006fbb49c8c`; its completed
checklist and provider-controls handoff are in
[TRACKER_INTEGRATION.md](https://github.com/freemocap/freemocap/blob/8e696f3ef3bf27a46de51343b5f987e6664c5ef1/TRACKER_INTEGRATION.md).
Both repository pushes were verified against `origin/sfn26`.

Validation: 98 Core tests plus seven subtests, eight Tracker schema/import tests,
16 full real-time replay cases and nine tensor/timing consumer tests passed.
Both full graph post-hoc recordings and the older pipeline's test-data run
passed. All six complete saved 2D channel tensors match the prior Core outputs
exactly; 192 controlled camera-frame comparisons also match exactly across
old/new Core factories. Reference tests used numerical outputs and recorded
frames, not Blender or hardware-camera/GUI acceptance.

The previous standalone Tracker baseline uses OpenCV 5.0.0 while Core uses
4.12.0; identical decoded pixels yielded different preprocessing tensors.
Cross-environment numerical comparisons must control this difference. No
packages or lockfiles changed. The Core research edits in `references.bib`
remain untouched and uncommitted. Global 3 has not started; its proposed checklist
is in Core's published handoff and awaits approval.

## Execution and PR handoff instructions

Reprint the complete active subtask checklist in chat after each completed step.
After completing a global stage, reprint the global checklist and expand the next
stage for approval. Completed subtask expansions do not need repeating afterward.
Pause at each completed PR/branch integration handoff and supply a short comment
the owner can copy into the originating PR. Never post that comment automatically.
Only say a PR was merged after verifying its Git ancestry; a selective
source port must be described as a port, with omitted work stated explicitly.
PR 81 includes PR 84's history. Merge PR 84 first, then its descendant PR 81,
then PR 90, preserving the complete histories and resolving compatibility issues.
Dependency installations are not authorized by this stage. The owner's latest
approval includes Core integration tests and their publication; Core runtime
migration remains Global 2. Existing source values, confidence semantics, landmark
order, download locks, shared-session ownership and dense numerical storage are
acceptance constraints.

## Pinned inputs and baseline

- Tracker baseline: `b535c0f07044e2922fa86cb50d184f46898386c8`.
- Core consumer: `1c104e93cdd2e24c18d90e3750ef3bc506a0b76f`.
- PR 84: `069ef01ffcf754ea05596c002071cd7858f2f8e7`.
- PR 81: `6560685cfe9411a27f5a7ca6192f91cc4ac506c3`.
- PR 90: `46d3454fd17d504f7d406d04f87497a2af1bbd2a`.

Private diagnostics live in the outer workspace's ignored
`.test-artifacts/sidecar-integration/`. `reference_baseline.py` saves names,
coordinates and confidence tensors for the first 12 frames of both existing
test and sample reference recordings, for one and three cameras, with explicit
CUDA execution and default RTMW/YOLOX models. Warm timings exclude the first two
frames; these short captures are regression evidence, not full-app benchmarks.
Each run confirms ONNX sessions are released on tracker close.
The initial shared-session, cache, provider and batch unit baseline passed 26 tests.
An initial pytest temporary-parent setup error was corrected before the passing run.
