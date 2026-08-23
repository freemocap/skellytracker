# Implement the Model Sidecar Spec + Dogfood YOLOX/RTMW

> Status as of commit `dd10819` (branch `philip/sidecar_implementation`): **M1, M2, and M3 are done** (M3 uncommitted in the working tree as of this update). M4–M5 (RTMW migration + cleanup) have not been started. See the "Status" line under each milestone below.

## Context

`specs/sidecar-spec.md` defines a YAML "sidecar" format that describes the runtime I/O contract for ONNX detector/pose models: input tensor shape, normalization, resize, batching, output semantics, and decode logic. Two commits (`ec12b32`, `837bd09`) landed the spec document itself; this plan implements it in code.

The goal is to "eat our own dog food": implement the sidecar system, then migrate the two existing hand-written ONNX detectors — YOLOX (person object detector) and RTMW/RTMPose wholebody (133-keypoint pose estimator) — to be driven by sidecar YAML instead of hardcoded Python constants (model URLs, mean/std, input sizes, output permutations). This must be a **no-functional-change** migration: same models, same math, same detection/keypoint output — just loaded declaratively.

Explicitly out of scope for this repo/plan: the skellycam API surface that exposes available models to FMC, and the FMC pipeline-builder UI. Those live in other repos.

Note: `CLAUDE.md` at the repo root is stale — it describes a pre-rewrite architecture (`trackers/`, `TrackedObjectDefinition`, `composite_gpu_tracker`) that no longer exists after commit `2c2615c` ("Structural rewrite (#75)"). This plan targets the **current** `skellytracker/core/` package. Refreshing `CLAUDE.md` is a flagged follow-up, not part of this plan.

## Key decisions

1. **Storage layout.** Sidecar YAML files are git-tracked inside the package tree at `skellytracker/core/detectors/object_detectors/{family}/{model_id}.yaml` and `skellytracker/core/detectors/keypoint_detectors/{family}/{part}/{model_id}.yaml`, matching the spec's layout exactly. ONNX binaries are **not** git-tracked — they still download through the existing `ModelSource`/`resolve_model_path()` (`skellytracker/core/sessions/model_registry.py`), extended to (a) save under the sidecar's own directory instead of the flat `~/.cache/skellytracker/models/`, (b) honor the sidecar's `filename` field instead of deriving it from the URL, and (c) verify `url_sha256`. This reuses the existing download/zip-extraction machinery rather than building a new one.

2. **`tracked_points` ordering for RTMW.** The existing `rtmpose_wholebody.yaml` point list is in a *re-ordered* "schema order" (body, right_hand, left_hand, face), requiring the hardcoded `_MODEL_TO_SCHEMA_PERM` permutation array to fix up the model's true native output order (body, face, left_hand, right_hand). The spec requires `tracked_points[i]` to equal the model's i-th native output index, so the new sidecar's `tracked_points` will be authored in **true native model order**, eliminating the permutation entirely. This changes the *array order* of the returned `Keypoints`, but not the name→value mapping (`Keypoints` is name-indexed; nothing in the codebase relies on positional order — confirmed via grep). Parity tests must therefore compare by name, not raw array equality.

3. **Detector replacement strategy.** Modify `YoloxPersonDetector` and `RTMPoseKeypointDetector`/`RTMWWholebodyDetector` *in place* — same registered `detector_type` string (`"yolox_person"`, `"rtmpose"`), same config field surface, same `detect`/`preprocess`/`postprocess`/`connections` signatures — so `build_object_detector`/`build_keypoint_detector`, `TrackerConfig`, and all existing call sites keep working unmodified. Internals swap hardcoded constants for sidecar-driven lookups, but decode math is reused verbatim from the existing pure functions in `yolox_preprocessing.py` / `rtmpose_preprocessing.py`.

4. **RTMW file relocation.** Move the wholebody detector from `keypoint_detectors/rtmpose/wholebody/` to a new `keypoint_detectors/rtmw/wholebody/` directory, matching the spec's own family naming (`model_id: rtmw-wholebody`). A grep of all importers found the following that must keep working: `skellytracker/core/detectors/keypoint_detectors/rtmpose/__init__.py`, `.../rtmpose/rtmpose_keypoint_detector.py` (already a re-export shim), `.../rtmpose/wholebody/run_demo.py`, plus test/example files importing `from skellytracker.core.detectors.keypoint_detectors.rtmpose import RTMPoseKeypointDetector` or directly from `rtmpose.wholebody.rtmpose_wholebody_detector`. Keep `rtmpose/__init__.py` and a `rtmpose/wholebody/rtmpose_wholebody_detector.py` re-export shim pointing at the new `rtmw/wholebody/rtmw_wholebody_detector.py` module so none of these break. `KEYPOINT_DETECTOR_REGISTRY["rtmpose"]` key is unchanged.

5. **YOLOX output-contract representation.** YOLOX's post-graph-surgery ONNX actually emits two tensors (pre-NMS boxes + scores), which doesn't fit the spec's single-tensor `detections`+`fields` model cleanly. Resolve by having the sidecar describe the **native, un-stripped checkpoint contract** (one `detections` output, `fields: [x1,y1,x2,y2,score]`, `requires_nms: false`) — the upstream OpenMMLab export's actual shape. The `prepare_yolox_onnx` graph-surgery step (dynamic batch + NMS-stripping optimization) stays exactly as-is, as an internal implementation detail not modeled in YAML, exactly as it isn't described by anything declarative today.

## Implementation

### M1 — Sidecar resolution engine + Pydantic model (no detector changes)

**Status: DONE (commit `dd10819`).**

New package `skellytracker/core/sidecar/`:

- `errors.py` — `SidecarError` base, plus `SidecarParseError`, `SidecarRefError` (missing file / cycle / sibling-key / escapes-cache_dir), `SidecarBaseError`, `SidecarSchemaVersionError`, `SidecarValidationError` (wraps Pydantic `ValidationError` with file path attached). Each error carries `.file_path` and names the offending directive, matching the spec's "recoverable catalog errors."
- `resolution.py` — pure YAML-merge logic, no Pydantic dependency:
  - `parse_sidecar_file(path) -> dict`
  - `resolve_sidecar_composition(path, cache_dir) -> dict` — resolves `$ref` for a file relative to *that file's own directory* before merging it into a `base` chain (order matters: ref-resolve top-down per file, then deep-merge bottom-up), enforcing "never escape above `cache_dir`" via `Path.resolve()` + `is_relative_to()`.
  - `_deep_merge(base, override)` — JSON Merge Patch semantics: mapping+mapping recurses key-by-key with `None`-value delete; sequence/scalar replaces wholesale. Used for both `base` inheritance and per-size `sizes.<name>` merging.
  - `_resolve_refs(node, current_file, cache_dir, stack)` — recursive `$ref` walker with cycle detection (`stack`) and strict-form enforcement (`{$ref: x, other: y}` errors).
- `model.py` — the `SidecarModel` Pydantic hierarchy (`extra="forbid"`), submodels: `InputSpec`, `CustomNormalization`, `ResizeSpec`, `CropPolicySpec`, `BatchingSpec`, `OutputSpec`, `DetectionDecodeSpec`, `PoseSpec`, `SkeletonSpec`, `PoseDecodeSpec`, `OverlaySpec`, `OverlayGroupSpec`, `SizeSpec`/`OnnxSpec`/`BatchArtifactGroup`/`PrecisionArtifact`. `SidecarModel` validates everything genuinely shared at the top level, but keeps `sizes: dict[str, dict[str, Any]]` as raw (unvalidated) dicts — `SidecarModel.resolved_size(name)` deep-merges `sizes[name]` over the top level (same `_deep_merge` as `base`) and validates the result as a `SizeSpec` where `input.shape`/`resize.target_size` become required.
  - `canonical_mapping`/`derived_points` `prefixes` + `[prefix]X: "Y"` expansion implemented as a shared, dependency-free helper (`core/io/canonical_mapping_expansion.py`), invoked as a `model_validator(mode="before")` on `PoseSpec`.
  - The spec's validation-rule checklist is mapped to field/model validators — single-field checks as `field_validator`s, single-submodel invariants as `model_validator`s on that submodel, and genuinely cross-cutting checks (role↔semantic↔decode consistency, `fields` must appear in `outputs`, overlay-vs-skeleton-vs-tracked_points) as `model_validator(mode="after")` on `SidecarModel` itself.
- `versioning.py` — `parse_skellytracker_version(v) -> (year, month, build, tag)`, `is_schema_version_supported(sidecar_version, installed_version) -> bool`, `require_stable_version(v)`, called from the loader.
- `loader.py` — `load_sidecar(path, cache_dir=None) -> SidecarModel`: `parse_sidecar_file` → `resolve_sidecar_composition` → `SidecarModel.model_validate()`, plus the `model_id == basename` check and `schema_version` support check.

Also done as part of M1: `skellytracker/core/io/canonical_mapping_expansion.py` (new, shared `prefixes`/`[prefix]` expansion helper) and `TrackerMapping.from_yaml` extended to use it.

Tests: `skellytracker/tests/sidecar/test_resolution.py` (18 tests, `tmp_path`-based fixtures covering `$ref` chains, cycles, `base`+null-delete, sibling-key errors, escape-cache_dir rejection) and `test_model_validation.py` (54 tests covering the validation-rule checklist, grouped by spec section). All passing; `ruff` clean.

### M2 — Storage/artifact plumbing

**Status: DONE (commit `dd10819`).**

- Extended `resolve_model_path()` in `skellytracker/core/sessions/model_registry.py` with optional `expected_filename` (save/lookup as `cache_dir / expected_filename` instead of deriving from the URL tail) and `expected_sha256` (verify downloaded bytes before caching, raise new `ModelIntegrityError` on mismatch — the corrupt temp file is deleted, nothing is written to the cache path). Additive/backward-compatible — existing callers unaffected (both default `None`).
- `skellytracker/core/sidecar/runtime.py`: `sidecar_model_spec(sidecar, size, batch_key, precision, name, sidecar_dir, prepare=None, coreml_options=None) -> OnnxModelSpec` (builds a lazy, pure `OnnxModelSpec` from the chosen artifact — `name` is caller-supplied, no I/O happens here; see the M3 follow-up note below for why this superseded an earlier eager version), `build_normalization_fn(input_spec, precision) -> Callable[[NDArray], NDArray]` (resolves `normalization_by_precision[precision] → normalization → "imagenet_bgr"` and returns the numpy transform for `none`/`unit_float`/`imagenet_bgr`/`imagenet_rgb`/`custom`), `resolve_normalization_mode(...)` helper.

Tests: `test_model_registry_sidecar_support.py` (sha256 verification and filename-override with mocked `requests.get`) and `test_runtime.py` (normalization dispatch, `OnnxModelSpec` construction from a local artifact) — no real downloads needed. All passing.

**Not yet built in M2** (deferred to M3/M4 since they're detector-specific): `sidecar_letterbox_preprocess`/`sidecar_detection_decode`/`sidecar_affine_crop_preprocess`/`sidecar_simcc_decode` dispatch functions in `runtime.py` — these wrap the existing preprocessing/decode functions and are added alongside each detector's migration.

### M3 — YOLOX migration

**Status: DONE (uncommitted on branch `philip/sidecar_implementation`, working tree).**

- New `skellytracker/core/detectors/object_detectors/yolox/yolox.yaml`: `role: [object_detector]`, `input.normalization: none` (ONNX graph normalizes internally, matching current behavior), `input.resize: {method: letterbox, preserve_aspect_ratio: true, pad_value: 114}`, `outputs` = single `detections` entry with `fields: [x1,y1,x2,y2,score]` (native contract, see decision 5), `decode: {box_format: xyxy, requires_nms: false, ...}` matching today's defaults. `sizes` keys kept identical to today's `model_name` strings (`yolox-tiny`, `yolox-m`). Real `url_sha256` computed for both checkpoint zips (downloaded once and hashed).
- Added `sidecar_letterbox_preprocess`/`sidecar_detection_decode` dispatch functions to `yolox_preprocessing.py` (**not** `runtime.py` as originally sketched — see below) — thin wrappers around the *existing* `yolox_letterbox_preprocess`, `_postprocess_prenms`, `_postprocess_yolox`, `multiclass_nms`, unchanged.
- **Deviation from the original sketch:** the plan's sketch had `runtime.py` importing from `yolox_person_detector.py`'s decode helpers while `yolox_person_detector.py` imports from `runtime.py` — a real circular import (confirmed by triggering it both ways depending on which module a test imports first). Fixed by (a) moving `_postprocess_prenms`/`_postprocess_yolox` into `yolox_preprocessing.py` (code unchanged), and (b) putting the two new `sidecar_*` dispatch wrappers in `yolox_preprocessing.py` too, which imports `build_normalization_fn` from `sidecar/runtime.py` — a one-directional dependency (`runtime.py` has zero knowledge of YOLOX). `yolox_person_detector.py` imports from both, no cycle.
- Captured a golden-output fixture from the **pre-migration** `YoloxPersonDetector` against the repo's existing test image, before touching the file: `skellytracker/tests/sidecar/fixtures/golden/yolox_m_detect.json` and `yolox_tiny_detect.json`.
- Rewrote `yolox_person_detector.py` internals: `_YOLOX_MODEL_URLS`/`_YOLOX_INPUT_SIZES` replaced by a module-level `_SIDECAR = load_sidecar(...)`; `preprocess`/`postprocess`/`detect` delegate to the sidecar + dispatch functions. `model_spec()` builds `OnnxModelSpec` directly from sidecar data rather than via `runtime.sidecar_model_spec()` — that helper eagerly downloads/verifies and renames the spec (`f"{model_id}-{size}"`), which would (a) trigger a network download merely by importing the module (`YOLOX_MODEL_SPECS` is built eagerly at import time) and (b) break the `spec.name == model_name` contract asserted by `TestYoloxPersonDetectorModelSpec`. Kept `OBJECT_DETECTOR_REGISTRY["yolox_person"]` and the config's public fields unchanged.
- **Follow-up (same working tree): `url_sha256` is now actually verified**, not just declared. `OnnxModelSpec` (`skellytracker/core/sessions/onnx_session.py`) gained two purely-additive optional fields, `expected_filename`/`expected_sha256`; `OnnxSession.create()`'s per-model loop threads `spec.expected_filename`/`spec.expected_sha256` into its (already-lazy, already-per-`create()`-call) `resolve_model_path(spec.source, ...)` call. Per user decision, the sidecar's declared filename (e.g. `yolox-m_b1_fp32.onnx`) is pinned as the cache filename instead of the old URL-tail-derived name.
- **Second follow-up (same working tree): `sidecar_model_spec()` redesigned to be the single, pure, generic builder** — the earlier YOLOX-specific hand-rolled `model_spec()` body (duplicating what `sidecar_model_spec()` does) was deleted; `YoloxPersonDetector.model_spec()` is now a direct call into `runtime.sidecar_model_spec(...)`. This was possible because the *original* reason the two diverged — `sidecar_model_spec()` eagerly downloading/verifying and renaming to `f"{model_id}-{size}"` — was itself a design flaw, not a fixed requirement: it conflated "build a declarative spec" with "resolve/cache/verify a model," work that already has a single home (`OnnxSession.create()`). Fixed by making `sidecar_model_spec()` pure (no I/O at all — not even checking a local artifact exists) and taking `name` as a required caller-supplied parameter instead of deriving it, since a fixed naming convention could collide across sidecars sharing one `OnnxSession`. `OnnxSession.create()` remains the only place resolution/verification happens for every sidecar-driven detector, present and future (e.g. M4's RTMW). No other detector calls `sidecar_model_spec()` yet, so this is a clean break with no other call sites to update.

Tests: `skellytracker/tests/sidecar/test_yolox_sidecar.py` (loads/validates the real file; parity tests comparing post-migration `detect()` output against the golden fixture via `np.allclose`, run for both `yolox-m` and `yolox-tiny` — both passed, exact match within tolerance). `test_yolox_detector.py` updated only to import `_postprocess_prenms`/`_postprocess_yolox` from their new home in `yolox_preprocessing.py`; all its assertions (including the `ValueError` message and `input_size` fallback contracts) pass unmodified. Full suite: `pytest skellytracker/tests` — 357 passed. `ruff`/`black`/`isort` clean on all touched files.

### M4 — RTMW migration

**Status: NOT STARTED.**

- New `skellytracker/core/detectors/shared/skeletons/coco133_tracked_points.yaml` and `coco133_skeletons.yaml` — ported from today's `rtmpose_wholebody.yaml`, reordered to native model order (body, face, left_hand, right_hand) per decision 2.
- New `skellytracker/core/detectors/keypoint_detectors/rtmw/wholebody/rtmw-wholebody.yaml` — `role: [pose_estimator]`, `input.resize: {method: affine_person_crop, crop_policy: {expand_ratio: 1.25, maintain_aspect_ratio: true}}`, `input.normalization: imagenet_bgr` (verified matching the existing hardcoded mean/std exactly), `pose.tracked_points`/`connections` via `$ref` to the shared fragments above, `pose.decode: {method: simcc}`, `overlay.groups` ported mechanically from today's `connection_groups()` prefix logic (`right_hand_`, `left_hand_`, `face_`, default `body`). `sizes` keys kept identical to today's `model_name` strings (`rtmw-l-m_256x192`, `rtmw-x-l_256x192`, `rtmw-x-l_384x288`). Compute real `url_sha256` for all three checkpoints (**requires downloading them once**).
- New `rtmw_wholebody_to_canonical_mapping.yaml` — the repo currently has **no** wholebody canonical mapping (only body/hand/mediapipe variants exist); compose the existing body + hand mappings using the spec's `prefixes: [right_hand_, left_hand_]` + `[prefix]` syntax (the `prefixes`-expansion machinery this needs was already built in M1). This is net-new functionality (today's detector applies no canonical mapping).
- Add `sidecar_affine_crop_preprocess`/`sidecar_simcc_decode` to `runtime.py` — wrap the existing `rtmpose_letterbox_preprocess`/`get_simcc_maximum`/unprojection math unchanged; decode zips `pose.tracked_points` directly against output rows (no permutation array).
- Move `rtmpose_wholebody_detector.py` → `skellytracker/core/detectors/keypoint_detectors/rtmw/wholebody/rtmw_wholebody_detector.py`; delete `_MODEL_TO_SCHEMA_PERM`; rewrite constants/`preprocess`/`postprocess`/`detect`/`model_spec` to be sidecar-driven, same pattern as YOLOX. Leave a re-export shim at the old `rtmpose/wholebody/rtmpose_wholebody_detector.py` path (confirmed needed — `test_run_batched.py` imports directly from it) and keep `rtmpose/__init__.py`'s re-exports pointing at the new location. `KEYPOINT_DETECTOR_REGISTRY["rtmpose"]` unchanged. `connection_groups()` becomes a thin wrapper reading `_SIDECAR.overlay.groups` (still used by `run_demo.py`/`demo_bbox_policy.py`).
- Capture golden fixture from pre-migration output (by-name comparison, since array order intentionally changes per decision 2), before touching the file.
- Delete the now-redundant plain `rtmpose_wholebody.yaml` schema file once the sidecar is validated and no longer referenced.

Tests: `test_rtmw_sidecar.py` (validates real file, asserts 133 tracked points, `canonical_mapping` prefix-expansion produces `right_hand_wrist`/`left_hand_wrist`, `overlay.groups` has 4 entries), parity test against golden fixture (compare by name, `np.allclose`). Confirm `test_run_batched.py`, `test_rtmpose_detectors.py`, `test_rtmpose_video.py` still pass unmodified against the shim.

### M5 — Cleanup

**Status: NOT STARTED.**

- Delete dead hardcoded constants once confirmed unused (`_YOLOX_MODEL_URLS`, `_INPUT_SIZES`, `_RTMPOSE_MEAN`/`_STD`, etc.).
- Run the full test suite (`pytest skellytracker/tests`) to catch anything transitively exercising YOLOX/RTMW through `Tracker`/`process_video`/`process_batch`.
- Flag (don't execute) a `CLAUDE.md` refresh describing the new `core/sidecar/` package as a follow-up.

## Verification

- `pytest skellytracker/tests` — full suite green, including the new `skellytracker/tests/sidecar/` directory and the existing YOLOX/RTMPose/run_batched tests unmodified.
- `ruff check skellytracker/` and `black`/`isort` clean.
- Manually run `python -m skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.run_demo` (moved demo) and the YOLOX demo path against a webcam or test video to visually confirm detections/keypoints look correct post-migration — this is the actual "no functional change" smoke test beyond numeric parity.
- Parity assertions (golden fixture vs. sidecar-driven output, `np.allclose` within tolerance) are the primary automated gate for "no functional change."

## Resuming this plan in a fresh session

M1/M2 code lives in commit `dd10819` (`build out sidecar infrastructure`) on `philip/sidecar_implementation`: `skellytracker/core/sidecar/`, `skellytracker/core/io/canonical_mapping_expansion.py`, extensions to `model_registry.py`/`tracker_mapping.py`, and `skellytracker/tests/sidecar/`. Start the next session at **M3 (YOLOX migration)** — it's the first milestone requiring real model downloads.
