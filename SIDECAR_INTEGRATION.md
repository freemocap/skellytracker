# Sidecar integration into sfn26

Approved 2026-10-31; revised by the owner to use full Git merges.
Runtime integration is limited to SkellyTracker in Global 1.
The owner explicitly authorized the Git changes needed for these merges, with
short commit messages and a pause after each PR for the owner to post a comment.
Core migration follows the completed Tracker integration.

## Approved global checklist

- [ ] **Global 1. Integrate the sidecar implementation and registry-based tracker factory into SkellyTracker `sfn26`, with verified compatibility and a reviewed Git handoff.**
- [ ] **Global 2. Migrate FreeMoCap’s model loading and tracker construction to the new SkellyTracker interfaces, then validate live and post-hoc processing against reference recordings.**
- [ ] **Global 3. Adapt Core PR #931’s execution-provider controls to the current API, UI, and shared inference service, then verify selection and failure behavior.**
- [ ] **Global 4. Implement and evaluate RTMW-body/MediaPipe-hand composition, and validate the custom-ONNX loading interface needed by the DLC work.**
- [ ] **Global 5. Benchmark supported precision, batch-size, and execution-provider combinations against the established baseline, then recommend which acceleration changes to adopt.**

## Approved Global 1 checklist

- [x] **1.1. Commit the existing SkellyTracker fixes and tests, and verify the three PR revisions and their merge order.**
- [x] **1.2. Merge PR #84 into `sfn26`, resolve compatibility issues, validate and push the result, then pause with its PR comment.**
- [x] **1.3. Merge PR #81 into `sfn26`, reconcile its complete sidecar contract, validate and push the result, then pause with its PR comment.**
- [ ] **1.4. Merge PR #90 into `sfn26`, reconcile its factory with shared-session ownership, validate and push the result, then pause with its PR comment.**
- [ ] **1.5. Validate the combined SkellyTracker changes against reference recordings and all affected detector interfaces.**
- [ ] **1.6. Verify the published SkellyTracker revision and prepare the FreeMoCap integration handoff.**

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
Paused for the owner's PR 81 comment; PR 90 has not been merged.

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
Dependency installations and changes to other repositories are not authorized
by this stage. Existing source values, confidence semantics, landmark
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
