# Model Sidecar Specification

## Changes in this revision (`vNEXT`)

> **For reviewers.** This revision extends the contract so that every model produced by
> `Roboflow-Exporter` (RF-DETR), `OpenMMLab-Exporter` (RTMDet, RTMW) and `Ultralytics-Exporter`
> (YOLO26 detection and pose) can be described. The evidence behind each change, and the per-repo
> findings, are in [`sidecar-spec-exporter-coverage-plan.md`](sidecar-spec-exporter-coverage-plan.md).
> `vNEXT` is a placeholder: replace it with the skellytracker release that ships this revision before
> merging. It appears in the examples, the changelog and the compatibility table.

### Summary of changes

| # | Area | Change | Driven by | Section |
|---|------|--------|-----------|---------|
| 1 | Resize | New `input.resize.method: stretch` (independent x/y resize, no pad). | RF-DETR | [`input.resize`](#inputresize) |
| 2 | Resize | Letterbox gains **required** `pad_position` (`top_left` \| `center`) and `rounding` (`half_even` \| `half_up`), with normative centre offsets and an exact-inverse unprojection. `pad_value` accepts a scalar or a per-channel array. | YOLO26 (centre, half-even), RTMDet (top-left, half-up) | [`input.resize`](#inputresize) |
| 3 | Resize | Normative unprojection for each method. All resizes are defined as **non-antialiased**. | all | [Unprojection](#unprojection) |
| 4 | Resize | `affine_person_crop` defines `maintain_aspect_ratio` and adds `crop_policy.pad_value`. | RTMW | [`input.resize`](#inputresize) |
| 5 | Input | New `input.color_format` (`rgb` \| `bgr`), required when normalization is channel-agnostic (`none`/`unit_float`). `none` now means "pixels unchanged in `input.dtype`" (not only uint8). The `imagenet_*` channel order is spelled out. **Behaviour change:** at `vNEXT`, `imagenet_bgr` applies BGR-ordered statistics (B mean `103.53`). Legacy sidecars keep the old meaning (the RGB-ordered tuple on BGR pixels, as rtmlib does). | YOLO26, RF-DETR, RTMDet | [`input`](#input), [Normalization modes](#normalization-modes) |
| 6 | Outputs | New detection semantics `boxes`, `scores`, `class_ids` and `class_logits`, with `class_axis`. Output dtypes gain `int32`, `int64` and `bool`. Output names must be unique. | RF-DETR (`dets` + `labels` logits), RTMDet (`dets` + int64 `labels`) | [Detection output semantics](#detection-output-semantics) |
| 7 | Decode | Detection `decode` is rebuilt around a required `selection` (`none` \| `top_k` \| `nms`), output bindings (`boxes_output`, `scores_output`, `class_ids_output`, `class_logits_output`), `box_scale`, `score_activation`, `class_id_map`, `clamp_boxes` and an `nms` block (`iou_threshold`, `score_threshold`, `class_agnostic`, `post_nms_top_k`). Stage order is normative, and score thresholds are strict (`>`). | RF-DETR (query×class top-k), RTMDet (host NMS), YOLO26 (graph-final rows) | [`decode`](#decode-detection), [Selection behaviour](#selection-behaviour) |
| 8 | Decode | `max_detections` no longer defaults to `300`. It now only declares the graph's fixed final row count, and is valid only with `selection: none`. `requires_nms` becomes a legacy alias of `selection`. `may_include_non_person_classes` is replaced by `is_person_only` (the old field inverted). | all detectors | [`decode`](#decode-detection) |
| 9 | Pose | New `packed_keypoints` on a `detections` output, for one-stage models that pack keypoints into the detection row. | YOLO26 pose (`[N,300,57]`) | [`packed_keypoints`](#packed_keypoints-bottom-up-packed-rows) |
| 10 | Pose | Bottom-up instance count is no longer overloaded onto `max_detections`. Pose rows are row-aligned with detection rows and gathered by the surviving detection indices. | YOLO26 pose, RTMO | [Bottom-up decode](#bottom-up-decode) |
| 11 | Sizes | **Family invariant:** a size entry may set only `input.shape`, `input.resize.target_size` and `onnx`. Sizes that differ in channel order, normalization or post-processing must be separate families. This resolves the earlier contradiction between "a size is a partial document deep-merged over the top level" and "sizes share the output contract". | RTMDet (person vs COCO vs RGB-backbone) | [`sizes`](#sizes) |
| 12 | Versioning | Explicit version-aware rules so that sidecars older than `vNEXT` keep loading with their old meaning. | existing YOLOX/RTMPose sidecars | [Version-aware compatibility](#version-aware-compatibility) |
| 13 | Examples | Reference examples replaced with real exports: YOLO26 detection, YOLO26 pose, RF-DETR, RTMDet person, RTMW. All batch keys are quoted, and the YOLO26 int8 input dtype is corrected to `float32`. | all | [Reference examples](#reference-examples) |

### Considered and deliberately not added

Each of these can be added later as a non-breaking schema bump when a model needs it.

| Field / feature | Reason |
|-----------------|--------|
| `resize.antialias`, `resize.size_divisor` | No in-scope model antialiases or has dynamic spatial dims. Resizes are normatively non-antialiased. |
| `crop_policy.border_mode`, rotated crops | Only constant borders and zero rotation are used. |
| `score_activation: softmax` | No model uses it, and how it behaves over gap/background slots is unresolved. |
| `class_id_map_before_selection` | No model maps classes before top-k. "Map after selection" is normative. |
| `nms.pre_nms_top_k`, `nms.min_bbox_size` | mmdet's `nms_pre` is per feature level, which a flat output can't express; a global top-k would describe a different algorithm. `min_bbox_size` is `0` everywhere. |
| `graph_candidate_limit` | Informational only; the host can't act on it. The RTMDet graph caps that motivated it were an export-config bug. Detection outputs are normatively the complete candidate set. |
| `pose.decode.instance_count` | The count is derivable from the detection row axis (change 10). |
| Soft-NMS / per-class IoU, masks, multi-input models, quantized outputs | Out of scope. No in-scope model needs them. |

### Compatibility impact

- Sidecars with `schema_version < vNEXT` (e.g. the in-repo YOLOX and RTMPose sidecars) keep loading unchanged under the [legacy rules](#version-aware-compatibility).
- Sidecars authored at `vNEXT` must declare `decode.selection`, letterbox `pad_position`/`rounding`, and `color_format` where required. They must not use `requires_nms` alone, `may_include_non_person_classes`, `letterbox` + `preserve_aspect_ratio: false`, or size-level overrides other than `input.shape`, `input.resize.target_size` and `onnx`.

---

## Overview

A **sidecar** is a YAML file that sits beside one or more ONNX model files and describes the runtime I/O contract a host must satisfy to load and run them correctly: input tensor, normalization, resize, batching, output semantics, and decode.

Sidecars support two model **roles**:

| Role | Purpose |
|------|---------|
| `object_detector` | Object detection (e.g. YOLO26, RF-DETR, RTMDet) — emits boxes, scores, and class IDs. |
| `pose_estimator` | Pose estimation (e.g. RTMW, RTMPose) — emits keypoint coordinates. |

A sidecar may declare **both** roles — e.g. a one-stage model such as RTMO or YOLO26-pose that emits boxes and keypoints in the same pass. In that case `role` lists both values, and `outputs` mixes detection and pose tensors (or packs keypoints into the detection rows; see [`packed_keypoints`](#packed_keypoints-bottom-up-packed-rows)).

### Core properties

- **One sidecar per model family.** A single `{model_id}.yaml` file lists every size of the model, and within each size every native batch size and precision variant, via `sizes.<size>.onnx.batch_artifacts`.
- **Multiple sizes per sidecar.** A family ships several sizes (e.g. `nano`, `small`, `medium`) in one file. All sizes share one preprocessing and output contract and differ only in input size and ONNX artifacts (see [`sizes`](#sizes)).
- **Traceable compatibility.** A sidecar's `schema_version` identifies the minimum skellytracker release required to consume it.
- **Single source of truth.** This document is the canonical, human-readable contract. It is **not parsed at runtime**; skellytracker mirrors it with Pydantic `BaseModel` subclasses (the `SidecarModel` hierarchy), and documentation and validators must stay aligned.
- **Composable.** A sidecar may `$ref` shared YAML fragments and may `base` another sidecar to inherit and override values.

### Document map

| Section | Purpose |
|---------|---------|
| [Format](#format) | File naming, style conventions, storage layout, and file-pairing rules. |
| [File composition](#file-composition-ref-includes-and-base-inheritance) | `$ref` includes and `base` inheritance for reuse. |
| [Common fields](#common-fields) | Shared identity, input, resize, batching, and output fields for every sidecar. |
| [Detection fields](#detection-fields-object_detector-role) | Output semantics, `decode`, and normative selection/NMS behaviour for the `object_detector` role. |
| [Pose-estimation fields](#pose-estimation-fields-pose_estimator-role) | Keypoints, skeletons, canonical mapping, packed keypoints, and `decode` for the `pose_estimator` role. |
| [Model sizes within a family](#model-sizes-within-a-family) | How multiple sizes share one contract. |
| [Normalization modes](#normalization-modes) | Pixel normalization options and channel order. |
| [Schema versioning](#schema-versioning) | How `schema_version` gates compatibility, including legacy rules. |
| [Validation rules](#validation-rules) | The complete validity checklist. |
| [Changelog](#changelog) | Contract change history keyed by `schema_version`. |
| [Reference examples](#reference-examples) | Complete annotated sidecar examples. |

## Format

### File extension and naming

Sidecars use the `.yaml` extension. The filename is **`{model_id}.yaml`** — the `model_id` field inside the file matches the basename.

| Artifact | Example |
|----------|---------|
| Sidecar (one per family) | `yolo26.yaml` |
| ONNX models (any size × batch × precision) | `yolo26-nano_b2_fp16.onnx`, `yolo26-small_b4_fp32.onnx` |

ONNX filenames encode size, batch, and precision: `{model_id}-{size}_{batch}_{precision}.onnx` (e.g. `yolo26-nano_b2_fp32.onnx`, where `{batch}` is `b2`). For a `dynamic` batch, the batch component is omitted: `{model_id}-{size}_{precision}.onnx` (e.g. `rtmw-wholebody-l-m_fp32.onnx`).

### Style conventions

- Document marker `---` at the top of file.
- `#` prefix for comments.
- YAML mappings and sequences — no trailing commas, no quotes on bare strings.
- Flat top-level keys; nested objects use indentation.
- String values quoted only when necessary.

### Storage layout

Sidecars and their ONNX files live under a model cache directory (`{cache_dir}`), which defaults to `skellytracker/core`. The layout mirrors the detector taxonomy:

```text
{cache_dir}/
  detectors/
    calibration_detectors/          # calibration targets (e.g. Charuco boards) — hard-coded, not sidecar-defined
    object_detectors/
      {family}/                     # role: object_detector
        {model_id}.yaml
        {model_id}-{size}_{batch}_{precision}.onnx
    keypoint_detectors/
      {family}/                     # role: pose_estimator
        hand/
          {model_id}.yaml
          {model_id}-{size}_{batch}_{precision}.onnx
        face/
          {model_id}.yaml
          ...
        body/
          {model_id}.yaml
          ...
        wholebody/
          {model_id}.yaml
          {model_id}-{size}_{batch}_{precision}.onnx
    shared/                         # optional $ref fragments (no schema_version)
      skeletons/                    # tracked-point lists and skeletons
      class_ids/                    # class_id_map fragments (e.g. coco91.yaml)
```

| Branch | Role | Contents |
|--------|------|----------|
| `detectors/calibration_detectors/` | calibration | Calibration detectors (e.g. Charuco boards). Hard-coded in skellytracker — **not** specified via sidecars. |
| `detectors/object_detectors/{family}/` | `object_detector` | One sidecar per family, e.g. `yolo26/`. |
| `detectors/keypoint_detectors/{family}/{part}/` | `pose_estimator` | Keypoint models per body region — `hand`, `face`, `body`, `wholebody`. |
| `detectors/shared/` | — | Shared `$ref` fragments. Optional; carry no `schema_version`. |

```text
detectors/object_detectors/yolo26/
  yolo26.yaml
  yolo26-nano_b2_fp32.onnx
  yolo26-nano_b2_fp16.onnx
  yolo26-small_b2_fp32.onnx

detectors/keypoint_detectors/rtmw/wholebody/
  rtmw-wholebody.yaml
  rtmw-wholebody-l-m_fp32.onnx
  rtmw-wholebody-x-l_fp32.onnx
```

### File pairing rules

- A sidecar and its ONNX files live together in the leaf model directory (see [Storage layout](#storage-layout)).
- **Fragment files** (shared sections pulled in via `$ref`) are plain YAML values — a mapping, sequence, or scalar — and carry **no** `schema_version`. Convention: keep them under `detectors/shared/`.
- **Base files** referenced by `base` are ordinary sidecars (`{model_id}.yaml`). A derived file that declares `base` is itself another `{model_id}.yaml` and must override `model_id` (and `schema_version`) so its basename matches its own `model_id`.

## File composition: `$ref` includes and `base` inheritance

Two directives cover reuse without duplication:

| Directive | Scope | Semantics |
|-----------|-------|-----------|
| `$ref` | any value, any depth | The value is **replaced** by the entire parsed content of the referenced file. Recursive. |
| `base` | top-level only | The file **inherits** from another sidecar; the current file's keys are **deep-merged over** it (current wins). Recursive. |

### `$ref` — reference another file for any value

A `$ref` directive is a YAML mapping with **exactly one key** named `$ref`, whose value is a file path:

```yaml
decode: {$ref: shared/yolo26_decode.yaml}
```

The resolver loads the referenced file and replaces the whole directive node with its parsed content, regardless of whether that content is a scalar, sequence, or mapping.

```yaml
# detectors/shared/skeletons/coco133_tracked_points.yaml  (a plain sequence)
- nose
- left_eye
- right_eye
# ... 133 names

# detectors/shared/skeletons/coco133_skeletons.yaml  (a plain sequence of named skeletons)
- name: rtmpose_skeleton
  edges:
    - [nose, left_eye]
    - [nose, right_eye]
- name: openpose_skeleton
  edges:
    - [nose, neck]
    - [neck, left_shoulder]

# detectors/keypoint_detectors/rtmw/wholebody/rtmw-wholebody.yaml  (excerpt — see the full RTMW reference example)
---
schema_version: "vNEXT"
model_id: rtmw-wholebody
display_name: RTMW WholeBody
role: [pose_estimator]
pose:
  estimator_type: top_down_single_person
  requires_object_detector: true
  tracked_points: {$ref: ../../../shared/skeletons/coco133_tracked_points.yaml}
  connections: {$ref: ../../../shared/skeletons/coco133_skeletons.yaml}
```

Rules:

- **Path resolution** — a `$ref` path resolves relative to the directory of the file that contains the directive (not the root sidecar). It may point anywhere under `{cache_dir}`, but must never escape above `{cache_dir}`.
- **Recursion** — an included file may itself contain `$ref` directives; they resolve relative to that file.
- **Cycle detection** — revisiting a path already in the resolution stack is an error naming the cycle.
- **Strict form** — a mapping containing `$ref` must have only `$ref`. `{$ref: x, other: y}` is an error.
- **No schema on fragments** — fragments do not carry `schema_version`; only the top-level sidecar is versioned and validated.
- **Reserved** — `$ref` is reserved and stripped during resolution; no validated field may be named `$ref`.

### `base` — inherit from another sidecar and override

A sidecar may declare `base` at the top level to start from an existing sidecar and override only what it changes:

```yaml
# detectors/object_detectors/yolo26-int8/yolo26-int8.yaml — a separately published int8 build of the family
# (illustrative: no current exporter writes derived files; this shows the mechanism)
---
schema_version: "vNEXT"
base: ../yolo26/yolo26.yaml
model_id: yolo26-int8
display_name: YOLO26 (int8)
sizes:
  nano:
    onnx:
      batch_artifacts:
        "1":
          precision_artifacts:
            int8:
              filename: yolo26-int8-nano_b1_int8.onnx
```

Everything not listed above (`input`, `batching`, `outputs`, `decode`, the other sizes, and the other precisions under `sizes.nano`) comes from `yolo26.yaml` unchanged.

**Merge semantics (deep merge, current wins):**

| Node type | Rule |
|-----------|------|
| mapping + mapping | Merge key-by-key; nested mappings recurse. Keys only in the base are kept; child keys replace base values. |
| mapping + sequence/scalar | Child replaces the base value entirely. |
| sequence + anything | Child replaces the base sequence entirely (sequences are **not** concatenated). |
| scalar + anything | Child replaces. |
| key absent in base | Child key is added. |
| key mapped to `null` | The key is removed from the merged result (delete). |

A child key mapped to `null` deletes that key from the merged result (JSON Merge Patch, RFC 7386) — the only way to remove an inherited key. `null` is reserved for deletion: no field uses `null` as a value; optional fields are omitted, never `null`. `null`-delete applies to `base` and `sizes` deep-merges, not to `$ref` (a node replacement).

```yaml
# Drop the int8 precision inherited from the base:
sizes:
  nano:
    onnx:
      batch_artifacts:
        "1":
          precision_artifacts:
            int8: null
```

Rules:

- **Top-level only** — `base` is recognized only at the document root.
- **Path resolution** — a `base` path resolves relative to the directory of the file that declares it; it may point anywhere under `{cache_dir}` and must never escape above `{cache_dir}`.
- **Chainable** — a base file may itself declare `base`; the chain resolves depth-first before merging.
- **Versioning** — after merging, the derived file's `schema_version` governs validation.
- **Reserved** — `base` is reserved at the top level and stripped during resolution.

### Resolution pipeline

Composition resolves **before** validation, so validation never sees `$ref` / `base`:

```text
parse_sidecar_file(path)            # yaml.safe_load → raw dict
  → resolve_sidecar_composition()   # resolve $ref includes + base inheritance → flat dict
  → SidecarModel.model_validate()   # typed model
```

Resolution errors are recoverable catalog errors naming the offending file and directive (missing file, parse failure in a fragment, cycle, `$ref` with sibling keys).

## Common fields

Fields shared by every sidecar regardless of `role`.

### Identity

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | yes | skellytracker release version the sidecar was authored against (see [Schema versioning](#schema-versioning)). Pattern: `vYYYY.0M.BUILD[-TAG]`. |
| `model_id` | string | yes | Model identifier; matches the sidecar basename. For object detectors it is the family (e.g. `yolo26`); for keypoint detectors it is `{family}-{part}` (e.g. `rtmw-wholebody`). |
| `display_name` | string | yes | Human-readable name. |
| `role` | array[enum] | yes | One or both of `object_detector` / `pose_estimator`, e.g. `[object_detector]`, `[pose_estimator]`, or `[object_detector, pose_estimator]` for a one-stage model such as RTMO or YOLO26-pose. |
| `sizes` | object | yes | Map of size name → size definition (see [Sizes](#sizes)). At least one size. |

### `sizes.<size>.onnx.batch_artifacts`

A mapping from **native batch size** → artifact group, declared **per size** under `sizes.<size>.onnx.batch_artifacts`. Keys are **strings**: either a positive integer batch size written as a quoted string, or the literal `dynamic`, which means the model accepts any runtime batch size. Each group contains a `precision_artifacts` map using the per-precision schema below.

> **YAML authors: quote integer batch keys.** The key set mixes integer-like batch sizes with the string `dynamic`, so `batch_artifacts` is a string-keyed map (`dict[str, ...]`). Write `"1":`, not `1:` — an unquoted key parses as a YAML integer and fails schema validation (`Input should be a valid string`).

The full input tensor shape for a group is `[N] + sizes.<size>.input.shape[1:]` — the batch key `N` supplies the batch dim, and the size's `input.shape` supplies the channels and spatial dims (identical across all precisions and batches for that size).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sizes.<size>.onnx.batch_artifacts` | object | yes per size | String-keyed map: quoted positive-integer native batch sizes (`"1"`, `"2"`, …) or a single `dynamic`. A size may list any number of batch keys. |
| `sizes.<size>.onnx.batch_artifacts.<N>.precision_artifacts` | object | yes per group | Map of `fp32` / `fp16` / `int8` → artifact descriptor for native batch `N`. |
| `sizes.<size>.onnx.batch_artifacts.<N>.output_shapes` | array[array] | when `len(batch_artifacts) > 1` | Output tensor shapes for native batch `N`, parallel to top-level `outputs` order. |

**Per-precision artifact fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `filename` | string | yes | ONNX filename relative to the sidecar's leaf model directory. |
| `url` | string | no | Download location. May point to an archive (e.g. `.zip`). |
| `url_sha256` | string | yes when `url` present | Lowercase hex SHA-256 of the bytes at `url` (the archive or file itself). |
| `input_dtype` | string | no | Optional; ONNX graph input element type for this artifact. Defaults to `input.dtype[precision]`; when authored, must agree with it. |

When the ONNX file is not present locally, the host may download it from `url` and must verify it against `url_sha256` — `url_sha256` is **mandatory** whenever `url` is present. `url` may point directly to an ONNX file, or to an archive; an archive may contain any number of files but must contain exactly one ONNX file, which is extracted and renamed to `filename`. `url` is optional — models that ship their ONNX files alongside the sidecar need no download location.

Rules:

- Each size's `batch_artifacts` must have at least one key; each key is a **string** — a quoted positive integer (`"1"`, `"2"`, …) or `dynamic`. Unquoted `1:` parses as an integer and is rejected.
- A size's `batch_artifacts` may contain integer-like keys **or** a single `dynamic` key — never both. Mixing `dynamic` with integer keys is a validation error.
- Each group must list at least one precision.
- Precision keys are the closed enum `fp32`, `fp16`, `int8`.
- `batching.native_batch_sizes` is **derived** per size by parsing the integer-like keys of `batch_artifacts` as integers. A `dynamic` key means any runtime batch size is accepted — it is not a separate field.
- A precision artifact's `input_dtype` (when authored) must agree with `input.dtype` for that precision.

### `input`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | ONNX input tensor name. |
| `dtype` | object | yes | Map of precision → ONNX graph input element type, e.g. `{fp32: float32, fp16: float16, int8: uint8}`. When all variants share one type, every entry is that type (e.g. `{fp32: float32, fp16: float32, int8: float32}`). Keys are a subset of `fp32`/`fp16`/`int8` covering the precisions declared in `batch_artifacts`. |
| `layout` | enum | no | `NCHW` (default) or `NHWC`. |
| `color_format` | enum | conditional | `rgb` or `bgr` — the channel order of the tensor the model consumes. **Required** when the normalization resolved for any declared precision is `none` or `unit_float` (those modes are channel-agnostic). Optional otherwise; when authored it must agree with `imagenet_rgb`/`imagenet_bgr` or with `custom.color_format`. |
| `normalization` | string or object | no | See [Normalization modes](#normalization-modes). Default `imagenet_bgr` (the RTMPose convention). |
| `normalization_by_precision` | object | no | Optional `fp32`/`fp16`/`int8` → named mode override. Values are mode strings only (no `custom` object); a `custom` normalization belongs at the top-level `normalization`. |
| `resize` | object | no | Resize config (see below); omitted only when the model consumes the source image unchanged (same effect as `method: none`). `target_size` is per-size (`sizes.<size>.input.resize.target_size`); other fields are shared. |

`shape` is **not** a top-level field — the input image shape is declared per size as `sizes.<size>.input.shape` (see [Sizes](#sizes)). Its batch axis is always `-1`: the concrete native batch size lives in the `batch_artifacts` keys, never in the shape. The channels and spatial dims are concrete, except that the spatial dims are `-1` when [`input.resize.supports_dynamic_size`](#inputresize) is `true`.

`color_format` is a single top-level value covering every precision. A family whose precisions would need different channel orders cannot be described and must be split.

`input` is required for `object_detector` and `pose_estimator` sidecars. This spec models **single-input** models only — `input` describes exactly one input tensor; multi-input models are out of scope.

Decoded box and keypoint coordinates are always in a **top-left origin** — `x` grows right, `y` grows down — in the model's input image; the host unprojects them to the source image (see [Unprojection](#unprojection)). There is no per-sidecar origin override.

### `input.resize`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `method` | enum | yes | `letterbox` (aspect-preserving resize + pad), `stretch` (independent x/y resize, no pad), `affine_person_crop` (top-down pose), or `none` (pass the source image unchanged — no resize). |
| `target_size` | array[2] | conditional | `[height, width]` the input is resized to. Declared in `sizes.<size>.input.resize.target_size`. Required when `method` is `letterbox`, `stretch`, or `affine_person_crop`, unless `supports_dynamic_size: true`. Omitted when `method: none`. |
| `supports_dynamic_size` | bool | no | Model accepts any input spatial size; `target_size` is omitted. Default `false`. |
| `pad_position` | enum | `letterbox`: yes | `top_left` or `center` — where the resized image sits on the padded canvas. `letterbox` only. |
| `rounding` | enum | `letterbox`: yes | How the resized width/height are rounded: `half_even` (Python `round`) or `half_up` (`floor(x + 0.5)`). `letterbox` only. |
| `pad_value` | int, float, or array[3] | no | Letterbox only; default `114`. Raw pixel value (before normalization). An array is per channel, in `input.color_format` order. |
| `preserve_aspect_ratio` | bool | no | Letterbox only; when present must be `true`. (`false` is a legacy alias for `stretch`; see [Version-aware compatibility](#version-aware-compatibility).) |
| `interpolation` | enum | no | Host resize filter; default `linear`. Applies to `letterbox`, `stretch`, and `affine_person_crop`. |
| `crop_policy` | object | no | `affine_person_crop` only; see below. |

When `supports_dynamic_size: true`, the model accepts any input spatial size: `target_size` is omitted and the size's `input.shape` spatial dims are `-1`. This applies to `letterbox` and `stretch` (the host resizes to the runtime-accepted size) and to `affine_person_crop` (the host warps the crop to the runtime-accepted size). When it is not set and `method` is `letterbox`, `stretch`, or `affine_person_crop`, `target_size` is required.

**Resizes are not antialiased.** Every interpolation value below names the plain OpenCV filter (`cv2.resize` / `cv2.warpAffine`, half-pixel centres) without a downscale-adaptive support. A host using another library must reproduce that behaviour (e.g. torchvision `antialias=False`, not PIL `BILINEAR`).

**`letterbox`**

The whole source image is resized with a single scale factor and padded to `target_size` `[T_h, T_w]`:

```text
s    = min(T_w / src_w, T_h / src_h)
n_w  = max(1, R(src_w * s))         # R = resize.rounding
n_h  = max(1, R(src_h * s))
dst  = full((T_h, T_w, 3), pad_value)
left, top = (0, 0)                                          # pad_position: top_left
left, top = (floor((T_w - n_w) / 2), floor((T_h - n_h) / 2))  # pad_position: center
dst[top:top+n_h, left:left+n_w] = resize(src, (n_w, n_h), <interpolation>)
```

With `center`, any odd remainder goes to the right/bottom. `half_even` with `center` matches Ultralytics `LetterBox`; `half_up` with `top_left` matches mmdet `Resize(keep_ratio=True)` + `Pad(size=…)`.

**`stretch`**

The whole source image is resized directly to `target_size`, with independent x and y scale factors; the host does not preserve aspect ratio, does not crop, and does not pad:

```text
dst = resize(src, (T_w, T_h), <interpolation>)
```

**`none`**

The model consumes the source image directly — no letterbox, no affine crop. `target_size` is omitted and the host passes the source image unchanged. The size's `input.shape` spatial dims are concrete and declare the exact resolution the model expects (the caller supplies frames at that size), unless `supports_dynamic_size: true`, in which case any spatial size is accepted.

**`affine_person_crop`**

The top-down pose convention: a person bounding box is warped to `target_size` by an affine transform (center + scale) rather than padding a full frame. Used by top-down models such as RTMW and RTMW3D. `crop_policy` controls how the source bounding box is expanded before the warp:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input.resize.crop_policy.expand_ratio` | float | no | BBox expansion factor before the affine warp; default `1.25`. |
| `input.resize.crop_policy.maintain_aspect_ratio` | bool | no | Default `true`. When `true`, the expanded box is enlarged along its shorter side to the target aspect ratio `T_w / T_h`; when `false`, it is warped directly, distorting aspect. |
| `input.resize.crop_policy.pad_value` | int, float, or array[3] | no | Constant border value for pixels outside the source image; default `0`. An array is per channel, in `input.color_format` order. |

Normative warp (rotation is always `0`; rotated crops are out of scope):

```text
center   = bbox center
(w, h)   = bbox size * expand_ratio
if maintain_aspect_ratio:
    if w > h * (T_w / T_h): h = w * (T_h / T_w)
    else:                   w = h * (T_w / T_h)
M        = affine map taking the (w, h) box around center onto [0, T_w] x [0, T_h]
dst      = warpAffine(src, M, (T_w, T_h), <interpolation>, borderMode=constant, borderValue=pad_value)
```

This matches mmpose `GetBBoxCenterScale(padding=1.25)` + `TopdownAffine`.

**Interpolation values:**

| Value | OpenCV constant | Typical use |
|-------|-----------------|-------------|
| `linear` | `cv2.INTER_LINEAR` | YOLO/Ultralytics, mmdet/mmpose, RF-DETR (default) |
| `area` | `cv2.INTER_AREA` | Downscaling with area resampling |
| `cubic` | `cv2.INTER_CUBIC` | Higher-quality upscale |
| `nearest` | `cv2.INTER_NEAREST` | Nearest-neighbor |

### Unprojection

Decoded coordinates are first converted to **model-input pixels** — `box_scale` / `coordinate_scale` `normalized_01` multiplies `x` by the model input width and `y` by its height; `normalized_11` maps `[-1, 1]` to `[0, width]` / `[0, height]` — and then mapped back to the source image:

| Method | Unprojection (model-input pixels → source pixels) |
|--------|---------------------------------------------------|
| `letterbox` | `x_src = (x - left) * src_w / n_w`, `y_src = (y - top) * src_h / n_h` — the exact inverse of the resize, using the rounded `n_w`/`n_h` and the pad offsets from the letterbox algorithm. |
| `stretch` | `x_src = x * src_w / T_w`, `y_src = y * src_h / T_h`. |
| `none` | Identity. |
| `affine_person_crop` | Apply the inverse of the affine warp `M`. |

### `sizes`

The top-level `sizes` map enumerates every scale of the model in a single sidecar.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sizes.<name>.input` | object | yes per size | Size-specific input. May contain **only** `shape` (required; batch axis always `-1`) and `resize.target_size` (required when `method` is `letterbox`/`stretch`/`affine_person_crop` and `supports_dynamic_size` is not set). |
| `sizes.<name>.onnx` | object | yes per size | Size-specific artifacts: `batch_artifacts` (required). |

A size entry may contain only `input` and `onnx` (with the sub-fields above). It is deep-merged over the top level using the same merge semantics as [`base`](#base--inherit-from-another-sidecar-and-override). Any other key in a size entry — `role`, `outputs`, `batching`, `decode`, `pose`, `overlay`, `input.normalization`, `input.color_format`, `input.resize.method`, and so on — is invalid. Size names are lowercase tokens such as `nano`, `small`, `medium`; hyphens are allowed (e.g. `l-m`, `x-l`).

**Family invariant.** A family has exactly one preprocessing and post-processing contract. If two sizes need a different channel order, normalization, resize method, output semantics, selection mode, NMS parameters, or class mapping, they belong to **different families** (different sidecars). For example, the RTMDet exports form three families: person-only, CSPNeXt COCO (BGR), and RGB-backbone COCO (ConvNeXt/Swin).

When `target_size` is declared, the size's `input.shape` spatial dimensions must equal it (for `NCHW`, `shape[2:4] == target_size`; for `NHWC`, `shape[1:3] == target_size`); with `supports_dynamic_size: true`, `target_size` is omitted. The first size listed is the default when no size is requested. How a host selects a size or precision at runtime is otherwise out of scope — this spec only declares what is available.

| Contract part | Where it lives |
|---------------|----------------|
| `role`, `outputs`, `input` (except `shape` / `resize.target_size`), `batching` | shared at top level |
| `decode` | top level, when `role` includes `object_detector` |
| `pose`, `overlay` | top level, when `role` includes `pose_estimator` |
| `input.shape`, `input.resize.target_size` (unless `supports_dynamic_size`), `onnx.batch_artifacts` | per size (`sizes.<size>`) |

### `batching`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `batch_axis` | int | no | Batch dimension axis; default `0`. Non-zero batch axes are out of scope. |
| `supports_dynamic_batch` | bool | derived | `true` when `batch_artifacts` contains a `dynamic` key. Derived at load time — not authored. |

> `batching.batch_conversion` is **not part of this spec** and is reserved for a future change set.

### `outputs` (common)

Each entry describes one ONNX output tensor. `outputs` is required for every sidecar. The fields below are common to every role; role-specific fields are described in the [object-detection](#detection-fields-object_detector-role) and [pose-estimation](#pose-estimation-fields-pose_estimator-role) sections.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | ONNX output tensor name. Unique within `outputs` — one ONNX tensor is declared exactly once. |
| `dtype` | object | yes | Map of precision → output element type, like `input.dtype` (e.g. `{fp32: float32, fp16: float32}`). May differ from the input precision. Values: `float32`, `float16`, `uint8`, `int8`, `int32`, `int64`, `bool`. |
| `shape` | array | no | Output tensor shape; `-1` marks a dynamic dim. Omitted for pose outputs whose shape varies with the input size (e.g. SIMCC bins). |
| `rank` | int | no | Tensor rank. Omitted when `shape` is omitted. |
| `semantic` | enum | yes | Closed set: `detections`, `boxes`, `scores`, `class_ids`, `class_logits`, `simcc_x`, `simcc_y`, `simcc_z`, `heatmap`, `keypoints`, `poses`. Links the tensor to `decode` / `pose.decode`. |

All sizes in a sidecar share one output contract: each output tensor has the same name, semantics, and keypoint/class count across sizes. Fixed-shape outputs differ only in the batch dimension; spatial-dependent pose outputs (SIMCC bins, heatmap H×W) may scale their spatial dims with the input size. Per-batch shapes are declared via `batch_artifacts.<N>.output_shapes`.

A sidecar with both roles lists every tensor in one `outputs` array. Each tensor's `semantic` routes it to its decoder: `detections`, `boxes`, `scores`, `class_ids`, and `class_logits` → `decode` (detection); `simcc_x`/`simcc_y`/`simcc_z`/`heatmap`/`keypoints`/`poses` → `pose.decode`. The detection semantics require `role` to include `object_detector`; the pose semantics require `role` to include `pose_estimator`.

```yaml
# One-stage detector + pose model with separate tensors (role: [object_detector, pose_estimator]).
# Illustrative, RTMO-style excerpt: no in-scope exporter emits this layout today. For the packed
# single-tensor layout that YOLO26-pose emits, see the YOLO26 pose reference example.
outputs:
  - name: dets
    dtype:
      fp32: float32
    shape: [-1, 100, 5]
    rank: 3
    semantic: detections
    fields: [x1, y1, x2, y2, score]
  - name: keypoints
    dtype:
      fp32: float32
    shape: [-1, 100, 17, 3]
    rank: 4
    semantic: poses
    keypoint_axis: 2

decode:                # detection decode — applies to the primary detection tensor
  box_format: xyxy
  score_field: score
  is_person_only: true
  selection: nms
  nms:                 # illustrative values; take them from the model's reference config
    iou_threshold: 0.45
    score_threshold: 0.1
    class_agnostic: true

pose:                  # pose decode — `poses` rows follow the surviving detection rows
  estimator_type: bottom_up
  requires_object_detector: false
  # ... tracked_points / connections ...
  decode:
    method: coordinate
```

## Detection fields (`object_detector` role)

Fields that apply when `role` includes `object_detector`.

### Detection output semantics

An `object_detector` sidecar declares exactly **one primary detection output** — either one `detections` or one `boxes` tensor — plus optional companion tensors:

| Semantic | Shape | Dtype | Meaning |
|----------|-------|-------|---------|
| `detections` | `[B, Q, F]` | float | Packed rows. `fields` names each of the `F` columns (e.g. `x1, y1, x2, y2, score, class_id`); may also carry [`packed_keypoints`](#packed_keypoints-bottom-up-packed-rows). Primary. |
| `boxes` | `[B, Q, 4]` | float | Box coordinates only. Scores come from a `scores` or `class_logits` companion. Primary. |
| `scores` | `[B, Q]` | float | Per-row confidence, row-aligned with the primary output. |
| `class_ids` | `[B, Q]` | `int32`/`int64` | Per-row model class index, row-aligned with the primary output. |
| `class_logits` | `[B, Q, C]` | float | Per-row, per-class logits or scores; class axis last unless `class_axis` says otherwise. |

`B` is the batch, `Q` the candidate/row count, `C` the model's class-slot count. Rules:

- At most one each of `scores`, `class_ids`, and `class_logits`; `class_ids` and `class_logits` are mutually exclusive.
- `scores` and `class_ids` shapes equal the primary output shape without its final dimension (for `dets: [B, Q, 5]`, `labels: [B, Q]`).
- `class_logits` requires a `boxes` primary output and `decode.selection: top_k`; its non-class axes equal the `boxes` outer axes `[B, Q]`. `C` must be concrete.
- `class_ids` values are model class indices, mapped to emitted ids by `decode.class_id_base` / `decode.class_id_map`.

### `outputs` (detection)

Additional fields on detection output tensors:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `fields` | array | conditional | Ordered field names per detection row (e.g. `[x1, y1, x2, y2, score, class_id]`). Required for `semantic: detections`, forbidden for the other detection semantics. Its length equals the last tensor dimension, unless `packed_keypoints` is present, in which case it equals `packed_keypoints.offset`. |
| `class_axis` | int | no | `class_logits` only: axis holding the class dimension; default `-1`. Negative values follow Python/ONNX indexing; it must not be the batch axis. |
| `packed_keypoints` | object | no | `detections` only, pose-capable one-stage models: see [`packed_keypoints`](#packed_keypoints-bottom-up-packed-rows). |

### `decode` (detection)

Required when `role` includes `object_detector`. Describes how the host turns the detection outputs into final detections in the source image.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `selection` | enum | yes | `none`, `top_k`, or `nms` — how candidate rows become final detections (see [Selection behaviour](#selection-behaviour)). |
| `boxes_output` | string | no | Name of the primary `detections`/`boxes` output; defaults to the sidecar's single primary detection output. |
| `scores_output` | string | no | Name of the `scores` output. Mutually exclusive with `score_field` and `class_logits_output`. |
| `class_ids_output` | string | no | Name of the `class_ids` output. Mutually exclusive with `class_field` and `class_logits_output`. |
| `class_logits_output` | string | conditional | Name of the `class_logits` output. Required for `selection: top_k`; invalid otherwise. |
| `box_format` | enum | no | `xyxy` (default), `xywh` (top-left corner + size), or `cxcywh` (centre + size). Applies to `fields[0:4]` of a `detections` output, or to the four columns of a `boxes` output. |
| `box_scale` | enum | no | `pixel` (default; model-input pixels), `normalized_01`, or `normalized_11` (see [Unprojection](#unprojection)). |
| `score_field` | string | conditional | Field holding per-row score; must appear in `fields`. Required for a `detections` primary output unless `scores_output` is used. Invalid for a `boxes` primary output. |
| `class_field` | string | no | Field holding per-row class index; must appear in `fields`. Invalid for a `boxes` primary output. |
| `score_activation` | enum | no | `none` (default) or `sigmoid`. Applied to `class_logits` before selection. |
| `class_id_base` | int | no | Emitted class id = `class_id_base + model class index`; default `0`. Ignored when `class_id_map` is present. |
| `class_id_map` | array[int] or `$ref` | no | Length `C`: index = model class index, value = emitted class id, `-1` = ignore (candidates with that class are dropped). Takes precedence over `class_id_base`. |
| `top_k` | int | conditional | Required for `selection: top_k`: number of (row, class) pairs to select. |
| `nms` | object | conditional | Required for `selection: nms`; invalid otherwise. See below. |
| `max_detections` | int | no | Only with `selection: none`: the fixed number of final rows the graph emits (e.g. `300` for YOLO26). Never a host-side cap. No default. |
| `is_person_only` | bool | no | Default `false`. When `true`, the model detects only people: candidates without a class source are assigned `person_class_id`, and candidates whose resolved class is not `person_class_id` are dropped. |
| `person_class_id` | int | no | Emitted class id of "person"; default `0`. With `class_id_map`, it must be one of the map's non-negative values. |
| `confidence_threshold_default` | float | no | Default `0.7`. The host's default final filter, applied after selection (strict `score > threshold`). Host policy — callers may override it. |
| `clamp_boxes` | bool | no | Default `true`. Clamp unprojected boxes to the source-image bounds. |
| `requires_nms` | bool | no | Legacy alias: when present it must agree with `selection` (`true` iff `selection: nms`). |

**`nms` object:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `iou_threshold` | float | yes | A candidate is suppressed when its IoU with a higher-scoring kept box exceeds this value. |
| `score_threshold` | float | yes | Candidate filter applied before NMS (strict `score > score_threshold`). Independent of `confidence_threshold_default`. |
| `class_agnostic` | bool | no | Default `false` (per-class NMS). With no class source every candidate has one class, so both settings are equivalent. |
| `post_nms_top_k` | int | no | Keep at most this many survivors per image, highest score first. Omit to keep all survivors. |

The detection outputs are the **complete candidate set** for the declared `selection`: the host cannot recover candidates a graph dropped. A graph may pre-filter at `nms.score_threshold` without declaring it, since the result is identical.

### Selection behaviour

All score comparisons are strict (`>`). Ties in any score ordering are broken by ascending row index (ascending flattened index for `top_k`). For each batch item:

**`selection: none`** — the primary output rows are already final detections (e.g. YOLO26's NMS-free head).

1. Resolve each row's score (`score_field` or `scores_output`) and class index (`class_field` or `class_ids_output`; none → no class source).
2. Map class indices via `class_id_map` (drop `-1`) or `class_id_base`; apply `is_person_only`.
3. Convert `box_format` to `xyxy`, apply `box_scale`, unproject, and clamp when `clamp_boxes`.
4. Apply `confidence_threshold_default`. Do not run NMS; the host does not pad or truncate to `max_detections`.

**`selection: top_k`** — per-row class logits with a global (row, class) top-k (e.g. RF-DETR).

1. `p = score_activation(class_logits)`; move `class_axis` last, so `p` is `[Q, C]`.
2. Flatten row-major: `flat = row * C + class`.
3. Stable sort descending by score; take the first `min(top_k, Q * C)` entries.
4. `row = flat // C`, `class = flat % C`; gather `boxes[row]`; score is `p[row, class]`.
5. Map `class` via `class_id_map` (drop `-1`, so the final count may be below `top_k`) or `class_id_base`; apply `is_person_only`.
6. Convert boxes, apply `box_scale`, unproject, clamp.
7. Apply `confidence_threshold_default`. Do not run NMS. Several selected pairs may share one row (one box, several classes); this is allowed.

**`selection: nms`** — the primary output rows are pre-NMS candidates (e.g. RTMDet).

1. Resolve scores; drop candidates with `score <= nms.score_threshold`.
2. Convert `box_format` to `xyxy` and apply `box_scale` (model-input pixels). Do not unproject yet.
3. Resolve and map class ids (drop `-1`); with no class source assign `person_class_id`; apply `is_person_only`.
4. Run hard NMS at `nms.iou_threshold` — per class unless `class_agnostic: true` — in model-input coordinates.
5. Sort survivors by descending score; keep the first `nms.post_nms_top_k` when set.
6. Unproject and clamp.
7. Apply `confidence_threshold_default`.

For a model with both roles, every step that drops or keeps a detection row drops or keeps the corresponding pose row (see [Bottom-up decode](#bottom-up-decode)).

## Pose-estimation fields (`pose_estimator` role)

Fields that apply when `role` includes `pose_estimator`.

### `outputs` (pose)

Additional fields on pose output tensors:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `keypoint_axis` | int | conditional | Axis holding the keypoints in a pose output. Default `1` for `(N, K, …)` tensors (`simcc_x`/`simcc_y`/`simcc_z`/`heatmap`/`keypoints`); required (`2`) for `poses` `(B, I, K, …)`. |
| `keypoint_count` | int | no | Number of keypoints in a pose output. When authored, must equal `len(tracked_points)`; when omitted, it is derived as `len(tracked_points)`. |

For `semantic: poses` (bottom-up, separate tensor), the tensor is `(B, I, K, 3)` — batch, instances (`I`), keypoints (`K`), and `(x, y, score)` per keypoint — with `keypoint_axis: 2`. When `pose.decode.is_3d: true`, it is `(B, I, K, 4)` — `x, y, z, score` per keypoint, with `z` root-relative depth in `pose.decode.depth_unit`. The instance axis `I` equals the primary detection output's row axis `Q`.

### `packed_keypoints` (bottom-up packed rows)

Some one-stage models append the keypoints to each detection row instead of emitting a separate `poses` tensor (e.g. YOLO26-pose: `[B, 300, 57]` = 6 detection columns + 17 × `(x, y, score)`). Such a tensor is declared **once**, as `semantic: detections`, with a `packed_keypoints` object:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `packed_keypoints.offset` | int | yes | Index of the first keypoint column; equals `len(fields)`. |
| `packed_keypoints.components` | array[enum] | yes | Per-keypoint column order, drawn from `x`, `y`, `z`, `score`. Must contain `x` and `y`; contains `z` iff `pose.decode.is_3d: true`. |

```yaml
outputs:
  - name: output0
    dtype: {fp32: float32, fp16: float32, int8: float32}
    shape: [-1, 300, 57]
    rank: 3
    semantic: detections
    fields: [x1, y1, x2, y2, score, class_id]
    packed_keypoints:
      offset: 6
      components: [x, y, score]
```

Rules:

- Valid only when `role` includes both roles, `pose.estimator_type: bottom_up`, and `pose.decode.method: coordinate`.
- `offset + K * len(components) == shape[-1]`, where `K = len(pose.tracked_points)`. Keypoint `k`'s component `c` is column `offset + k * len(components) + index(c)`.
- Keypoint coordinates are in `pose.decode.coordinate_scale`, which must equal `decode.box_scale`. A keypoint `score` is already activated.

### `pose` (pose estimator)

The pose output contract declares the keypoints a model produces, the skeleton(s) joining them, how they map onto the canonical (VRM 1.0) skeleton, and how raw outputs are decoded into keypoints.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pose.estimator_type` | enum | yes | `top_down_single_person`, `top_down_multi_person`, or `bottom_up`. Top-down models see one person per crop (single-person outputs); bottom-up models (e.g. RTMO, YOLO26-pose) emit all persons in one pass. |
| `pose.requires_object_detector` | bool | yes | `true` when the model consumes a person crop produced by an upstream detector (top-down models — the host warps each detection box via `affine_person_crop`); `false` when the model consumes the full image (bottom-up models). Must be consistent with `estimator_type`. |
| `pose.landmark_schema` | string | no | Keypoint convention the model uses (e.g. `COCO_WholeBody`, `COCO17`, `MediaPipe_Hand`, `MediaPipe_Pose`, `LaPa_106`). |
| `pose.tracked_points` | array[string] or `$ref` | yes | Ordered list of keypoint names the model outputs. The i-th name is the i-th model output index. |
| `pose.connections` | array[Skeleton] or `$ref` | yes | Named skeletons; each joins `tracked_points` into a different skeleton. |
| `pose.derived_points` | object | no | Optional map of derived point name → derivation. May be referenced by `connections` edges. |
| `pose.canonical_mapping` | object or `$ref` | no | Maps model keypoint names onto the canonical (VRM 1.0) skeleton (see [Canonical skeleton](#canonical-skeleton)). |
| `pose.decode` | object | yes | How raw model outputs are decoded into keypoints (see below). Required for all `pose_estimator` models. |

```yaml
pose:
  estimator_type: top_down_single_person
  requires_object_detector: true
  tracked_points:
    - nose
    - left_shoulder
    - right_shoulder
    # ...
  derived_points:
    neck: [left_shoulder, right_shoulder]
  connections:
    - name: rtmpose_skeleton
      edges:
        - [nose, left_shoulder]
        - [left_shoulder, right_shoulder]
        # ...
    - name: openpose_skeleton
      edges:
        - [nose, neck]
        - [neck, left_shoulder]
        # ...
  # $ref paths resolve relative to this sidecar's directory.
  canonical_mapping: {$ref: ../../rtmpose/body/rtmpose_body_to_canonical_mapping.yaml}
  decode:
    method: simcc
```

#### `pose.tracked_points`

The complete ordered list of keypoint names produced by the model, **always in the model's output order** — the i-th name corresponds to the i-th row of the model's keypoint output. Names must be unique within a sidecar; side-specific points are authored literally (e.g. `right_hand_root`, `left_hand_root`). The keypoint count is derived: `len(tracked_points)`.

#### `pose.connections`

Named skeletons joining `tracked_points` entries. A sidecar may list more than one skeleton for the same keypoints — e.g. an RTMW wholebody output can be joined with an RTMPose skeleton or an OpenPose skeleton by selecting a different `connections` entry. The first entry is the default when no skeleton is named.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pose.connections[].name` | string | yes | Skeleton name, unique within the sidecar (e.g. `rtmpose_skeleton`, `openpose_skeleton`). |
| `pose.connections[].edges` | array[[name, name]] | yes | Edge pairs joining `tracked_points`/`derived_points` names. |

Both names in every edge must appear in `tracked_points` or `derived_points`.

#### `pose.derived_points`

Derived points are computed from `tracked_points` and may be referenced by `connections` edges (e.g. OpenPose's `neck`). Each entry uses the same forms as [`canonical_mapping`](#posecanonical_mapping), including the `prefixes` / `[prefix]` expansion:

| Form | YAML | Meaning |
|------|------|---------|
| passthrough | `nose_dup: "nose"` | derived = tracker keypoint at that name |
| mean | `neck: ["left_shoulder", "right_shoulder"]` | derived = unweighted mean of the listed keypoints |
| weighted | `mid_hip: {left_hip: 0.5, right_hip: 0.5}` | derived = weighted sum (weights normalised) |

Like `canonical_mapping`, a `prefixes` list expands one entry to both sides — the prefix applies to the derived name and to each source. With `prefixes: [right_hand_, left_hand_]`, `[prefix]palm_center: ["root", "thumb1", "forefinger1"]` produces `right_hand_palm_center = mean(right_hand_root, right_hand_thumb1, right_hand_forefinger1)` and `left_hand_palm_center = mean(left_hand_root, left_hand_thumb1, left_hand_forefinger1)`. Entries without `[prefix]` are literal.

#### `pose.canonical_mapping`

Maps the model's `tracked_points` onto the freemocap **canonical skeleton** — the [VRM 1.0](https://github.com/vrm-c/vrm-specification/tree/master/specification/VRMC_vrm-1.0) humanoid skeleton (see [Canonical skeleton](#canonical-skeleton)). Each entry keys a **canonical** landmark name to its tracker source: the key is the canonical landmark, and the value names the tracker keypoint(s) that produce it. Three mapping forms are supported, mirroring `TrackerMapping` in `skellytracker/core/io/tracker_mapping.py`:

| Form | YAML | Meaning |
|------|------|---------|
| passthrough | `left_elbow: "left_elbow"` | canonical = tracker keypoint at that name |
| mean | `hips_center: ["left_hip", "right_hip"]` | canonical = unweighted mean of the listed keypoints |
| weighted | `head_center: {left_ear: 0.5, right_ear: 0.5}` | canonical = weighted sum, normalised by the sum of the weights of the keypoints that are actually present |

A `prefixes` list makes one entry apply to multiple sides:

```yaml
canonical_mapping:
  prefixes: [right_hand_, left_hand_]
  "[prefix]wrist": "root"
  "[prefix]ring_finger_tip": "ring_finger4"
```

`[prefix]` keys must be YAML-quoted: an unquoted leading `[` opens a flow sequence, which is a parse error as a block-mapping key (see `rtmw/wholebody/rtmw_wholebody_to_canonical_mapping.yaml`).

`[prefix]X: "Y"` expands, for each `p` in `prefixes`, to canonical `pX` = tracker `pY`. So `[prefix]ring_finger_tip: "ring_finger4"` produces both `right_hand_ring_finger_tip = right_hand_ring_finger4` and `left_hand_ring_finger_tip = left_hand_ring_finger4`. Entries without `[prefix]` are literal.

**Missing-keypoint semantics.** A canonical landmark is produced only when its tracker source is present: a passthrough whose named keypoint is missing is omitted, and mean/weighted forms drop missing keypoints — a weighted sum is normalised by the sum of the weights of the *present* keypoints (not the declared total). This matches `TrackerMapping.apply()`.

Single-part canonical names are unprefixed (`wrist`, `thumb_cmc`); whole-body canonical skeletons with two hands use `prefixes` to produce prefixed names (`right_hand_wrist`, `left_hand_wrist`). `prefixes` apply only here in `canonical_mapping` — `tracked_points` are always authored as a complete, literal list. Note the distinction: `prefixes` is a sidecar authoring convenience, whereas `TrackerMapping` also accepts a runtime `prefix` that strips a prefix from tracker keypoint names before lookup (used to apply one hand mapping to both `right_hand_*` and `left_hand_*` keypoints).

#### Canonical skeleton

The canonical skeleton every tracker is mapped onto is the **VRM 1.0** humanoid skeleton. Its humanoid bones (per [`humanoid.md`](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_vrm-1.0/humanoid.md)) are:

| Group | Bones |
|-------|-------|
| torso | `hips`, `spine`, `chest`, `upperChest`, `neck` |
| head | `head`, `leftEye`, `rightEye`, `jaw` |
| legs | `leftUpperLeg`, `leftLowerLeg`, `leftFoot`, `leftToes`, `rightUpperLeg`, `rightLowerLeg`, `rightFoot`, `rightToes` |
| arms | `leftShoulder`, `leftUpperArm`, `leftLowerArm`, `leftHand`, `rightShoulder`, `rightUpperArm`, `rightLowerArm`, `rightHand` |
| fingers | `leftThumbMetacarpal`…`leftLittleDistal`, `rightThumbMetacarpal`…`rightLittleDistal` |

#### Reference implementations

The gold-standard reference for authoring `canonical_mapping` is the existing keypoint detectors. Each carries a tracker→canonical mapping YAML beside it, applied by `TrackerMapping` (`skellytracker/core/io/tracker_mapping.py`):

- `skellytracker/core/detectors/keypoint_detectors/rtmpose/body/rtmpose_body_to_canonical_mapping.yaml`
- `skellytracker/core/detectors/keypoint_detectors/rtmpose/hand/rtmpose_hand_to_canonical_mapping.yaml`
- `skellytracker/core/detectors/keypoint_detectors/mediapipe/body/mediapipe_body_to_canonical_mapping.yaml`
- `skellytracker/core/detectors/keypoint_detectors/mediapipe/hands/mediapipe_hand_to_canonical_mapping.yaml`

Their canonical landmark naming conventions:

| Region | Canonical names |
|--------|-----------------|
| body | COCO-WholeBody (`nose`, `left_shoulder`, `left_big_toe`, …) plus computed markers `head_center`, `neck_center`, `trunk_center`, `hips_center` |
| hand | MediaPipe Hand (`wrist`, `thumb_cmc`, `pinky_tip`, …) |
| face (standalone) | LaPa_106 (`face_0000`, …, 106 points) |
| face (wholebody) | iBUG 300-W 68-point (`face_0000`, …, `face_0067`) |

> **Name migration (current → VRM 1.0).** The names in this table are the **current** canonical landmark conventions used by the reference detectors. They will be renamed to the VRM 1.0 humanoid bones listed in [Canonical skeleton](#canonical-skeleton) (`hips`, `spine`, …, `leftHand`, `rightHand`, plus the finger bones). Until the rename lands, `canonical_mapping` keys use these current names; after the rename, they must use the VRM 1.0 names.

#### `pose.decode`

Declares how the model's raw output tensors become keypoints.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pose.decode.method` | enum | yes | `simcc`, `heatmap`, or `coordinate`. |
| `pose.decode.coordinate_scale` | enum | no | `coordinate` only: scale of the decoded coordinates — `pixel` (raw pixels in the input image; default), `normalized_01` ([0, 1] fraction of the input), or `normalized_11` ([-1, 1] centered on the input). |
| `pose.decode.is_3d` | bool | no | Default `false`. When `true`, the model emits a third coordinate: a `simcc_z` output (simcc), `(N, K, 3)` coordinates (`coordinate`), a `poses` tensor `(B, I, K, 4)`, or a `z` packed component (bottom-up `coordinate`). 3D `heatmap` is **not supported**. |
| `pose.decode.depth_unit` | enum | required when `is_3d: true` | Unit of the root-relative depth coordinate: `m` (RTMW3D), `mm`, or `pixel` (MediaPipe, image-width-scaled). |
| `pose.decode.depth_range` | array[2] | required when `is_3d: true` and `method: simcc` | `[min, max]` root-relative depth (in `depth_unit`) that `Z_bins` span; the decoded bin position maps linearly to depth. |

Decode methods:

- `simcc` — one `simcc_x` `(N, K, X_bins)` and one `simcc_y` `(N, K, Y_bins)` output (plus `simcc_z` `(N, K, Z_bins)` when `is_3d: true`). The bin counts are read from the output tensors; the bin→pixel scale is `input_width / X_bins` and `input_height / Y_bins` (see Shape key). Each keypoint's location is the argmax bin and its score is the max value. Decoded keypoints are in cropped image space; the host unprojects them to the source image. The `z` axis maps `Z_bins` linearly onto `depth_range` (in `depth_unit`) when `is_3d: true`.
- `heatmap` — one output with `semantic: heatmap`, shape `(N, K, H, W)` (2D only); per-keypoint peaks are decoded to locations in cropped image space, then unprojected to the source image.
- `coordinate` — coordinates are already decoded (no argmax) and live in the scale of the model's input image, per `coordinate_scale`; the host unprojects them to the source image. Three forms:
  - single-person — one output with `semantic: keypoints`, shape `(N, K, 2)` (`x, y`) or `(N, K, 3)` (`x, y, z`) when `is_3d: true`; `z` is root-relative depth (in `depth_unit`).
  - bottom-up, separate tensors — a primary detection output plus a `poses` output.
  - bottom-up, packed — a `detections` output carrying `packed_keypoints`.

#### Bottom-up decode

Bottom-up models (`bottom_up`, e.g. RTMO, YOLO26-pose) work on the full image and decode via `coordinate`. Each detection row `q` owns pose row `q` — either `poses[b, q]` or the packed keypoint columns of `detections[b, q]` — so the instance count always equals the detection row count. The host runs the detection `decode` (`selection: none` or `nms`), and emits keypoints for exactly the surviving detection rows, in the same order. Keypoints are unprojected with the same transform as the boxes.

Shape key:

- `N` — batch size (person crops for top-down models).
- `B`, `I` — batch and instance axes of a bottom-up `poses` tensor; `I` equals the detection row count `Q`.
- `K` — number of keypoints (`len(tracked_points)`).
- `H`, `W` — spatial height and width of the output tensor.
- `X_bins`, `Y_bins` — SIMCC x/y bin counts, read from the output tensors; the bin→pixel scale is `input_width / X_bins` and `input_height / Y_bins`.
- `Z_bins` — SIMCC depth bin count (`is_3d: true` only); bins span `pose.decode.depth_range` linearly.
- For `is_3d: true`, the third coordinate is root-relative depth (RTMW3D), in `depth_unit`. Note: `(N, K, 3)` is `x, y, z` for 3D single-person keypoints, `x, y, score` for bottom-up `poses` `(B, I, K, 3)`, and `(B, I, K, 4)` is `x, y, z, score` for 3D bottom-up `poses`.

### `overlay` (visualization, pose estimator)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `overlay.skeleton` | string or array[[name, name]] | no | Skeleton to draw: a `pose.connections[].name`, or an inline edge list. Defaults to the first named skeleton. |
| `overlay.groups` | object | no | Map of group name → group schema (below). Each group selects a subset of the skeleton's edges and colors them. |
| `overlay.keypoint_color` | array[3] | no | Default keypoint color `[r, g, b]`. |

**Group schema** (`overlay.groups.<name>`) — mirrors the RTMPose annotator's `ConnectionGroupSchema`:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `connections` | array[[name, name]] | no | Explicit edge subset of the skeleton (alternative to `prefix`). |
| `prefix` | string | no | Select skeleton edges where **either** endpoint starts with this prefix (e.g. `right_hand_`). |
| `connection_color` | array[3] | yes | RGB color for this group's edges. |
| `connection_thickness` | int | no | Default `2`. |
| `keypoint_color` | array[3] | no | RGB color for this group's keypoints; omit to use `overlay.keypoint_color`. |

Each group selects its edges via `connections` or `prefix` (not both). A `prefix` group matches every skeleton edge where **either** endpoint starts with the prefix; prefixed groups are evaluated in **declaration order**, and the first match wins. A group with neither `connections` nor `prefix` is the **default** group — it receives every skeleton edge not selected by another group (the RTMPose `body` group); at most one default group is allowed. Group names are free-form; the RTMPose convention is `body`, `right_hand`, `left_hand`, `face`.

## Model sizes within a family

A family ships several sizes of the same model — e.g. `yolo26` ships `nano`, `small`, and `medium` — all declared in a single `yolo26.yaml`. All sizes share `role`, `outputs` semantics, `batching`, the input tensor contract (`name`, `dtype`, `layout`, `color_format`, `normalization`, `resize` other than `target_size`), `decode`, and `pose`. Only `input.shape`, `input.resize.target_size`, and the ONNX artifacts are per size; spatial-dependent pose output dimensions may vary with the input size and are read at runtime.

```yaml
# Excerpt of yolo26.yaml (Ultralytics-Exporter): every YOLO26 size is exported at 640x640;
# sizes differ only in their artifacts. medium/large/xlarge follow the same pattern.
sizes:
  nano:
    input:
      shape: [-1, 3, 640, 640]
      resize:
        target_size: [640, 640]
    onnx:
      batch_artifacts:
        "1":
          precision_artifacts:
            fp32: {filename: yolo26-nano_b1_fp32.onnx}
            fp16: {filename: yolo26-nano_b1_fp16.onnx}
            int8: {filename: yolo26-nano_b1_int8.onnx}
  small:
    input:
      shape: [-1, 3, 640, 640]
      resize:
        target_size: [640, 640]
    onnx:
      batch_artifacts:
        "1":
          precision_artifacts:
            fp32: {filename: yolo26-small_b1_fp32.onnx}
            fp16: {filename: yolo26-small_b1_fp16.onnx}
            int8: {filename: yolo26-small_b1_int8.onnx}
```

Each size entry deep-merges over the shared top-level contract and may set only the per-size fields (see [Sizes](#sizes)). Sizes that would need any other difference are a separate family.

## Normalization modes

`input.normalization` declares how the host transforms resized pixels before `session.run()`. Use a closed enum string, or a `custom` object.

| Mode | Host behavior | Typical use |
|------|---------------|-------------|
| `none` | Pass pixel values (0–255) unchanged, cast to `input.dtype`. Channel order from `input.color_format`. | Graphs that normalize internally (e.g. YOLOX) |
| `unit_float` | `pixel / 255.0` per channel → `[0, 1]`. Channel order from `input.color_format`. | YOLO26 — Ultralytics-style float input |
| `imagenet_bgr` | `(pixel - mean) / std` on 0–255 pixels in **BGR** channel order: B uses mean `103.53` / std `57.375`, G `116.28` / `57.12`, R `123.675` / `58.395`. (Legacy sidecars use a different meaning; see [Version-aware compatibility](#version-aware-compatibility).) | RTMDet (CSPNeXt) |
| `imagenet_rgb` | `(pixel - mean) / std` on 0–255 pixels in **RGB** channel order: R uses mean `123.675` / std `58.395`, G `116.28` / `57.12`, B `103.53` / `57.375`. | RTMW, RF-DETR, RTMDet (ConvNeXt/Swin) |
| `custom` | `(pixel * scale - mean) / std` on 0–255 channel values | Rare exports that match no named mode |

Both `imagenet_*` modes use the same per-colour ImageNet statistics (`0.485/0.456/0.406` and `0.229/0.224/0.225` × 255); they differ only in the channel order of the tensor. Normalization is applied after the resize (after the affine warp for `affine_person_crop`; directly on the source image for `method: none`). `none` and `unit_float` are channel-agnostic, so `input.color_format` is required with them. `custom` declares its own channel order via `color_format`; when `custom.color_format` is omitted it inherits `input.color_format`, falling back to `rgb`.

**Named mode (preferred):**

```yaml
input:
  color_format: rgb
  normalization: unit_float
```

**Custom mode:**

```yaml
input:
  normalization:
    mode: custom
    color_format: rgb            # optional; inherits input.color_format, else rgb — order of mean/std
    scale: 0.00392156862745098   # optional; default 1.0
    mean: [0.0, 0.0, 0.0]
    std: [1.0, 1.0, 1.0]
```

**Per-precision overrides:**

```yaml
# For an int8 artifact whose graph takes raw uint8 pixels. The Ultralytics exporter emits this
# override only when an int8 artifact's input dtype is uint8/int8; the current YOLO26 int8
# exports take float32 input, so they need no override.
input:
  dtype: {fp32: float32, fp16: float32, int8: uint8}
  color_format: rgb
  normalization: unit_float
  normalization_by_precision:
    int8: none
```

Resolution order for a selected precision: `normalization_by_precision[precision]` → top-level `normalization` → default `imagenet_bgr`. `normalization_by_precision` values are named mode strings only; a `custom` normalization cannot be a per-precision override — declare it at the top-level `normalization` instead.

> ⚠️ The `imagenet_bgr` default is the RTMPose convention. Detector models must declare their normalization explicitly — relying on the default yields incorrect normalization for most detectors.

**Validation rules:**

- `normalization` defaults to `imagenet_bgr` when omitted.
- String values must be one of `none`, `unit_float`, `imagenet_bgr`, `imagenet_rgb`.
- Object form must have `mode: custom`; `mean`/`std` are length-3 sequences when present; `scale` is a positive number when present; `color_format` (when present) is `rgb` or `bgr`.
- `normalization_by_precision` keys must be a subset of `fp32`, `fp16`, `int8`; values must be valid mode strings.
- `input.color_format` is present when any declared precision resolves to `none` or `unit_float`, and (when present) agrees with `imagenet_rgb` (`rgb`), `imagenet_bgr` (`bgr`), and `custom.color_format`.

## Schema versioning

`schema_version` is a **string** matching the skellytracker release version that introduced or last required a change to the sidecar contract.

| Aspect | Rule |
|--------|------|
| Field | `schema_version` |
| Type | string |
| Value | Exact skellytracker release version (e.g. `"v2024.09.1019"`) |
| Pattern | `vYYYY.0M.BUILD[-TAG]` |

### Semantics

- **`schema_version` on a sidecar** = the skellytracker version that defined the contract the sidecar was authored against.
- **Minimum consumer** — skellytracker release `S` loads a sidecar with `schema_version` `V` when `S >= V`. Older sidecars (lower `schema_version`) remain supported by newer skellytracker releases unless a release note says otherwise; newer sidecars (higher `schema_version`) are rejected by older skellytracker releases.
- **Too-new sidecar** — if `sidecar.schema_version > installed_skellytracker_version`, reject with a recoverable error naming both versions and an upgrade hint.
- **Exporter obligation** — set `schema_version` to the skellytracker version documented as current at export time.
- **Stable versions only** — sidecars must not use pre-release tags (`-TAG` suffix).

### Version comparison

`parse_skellytracker_version(version) -> (year, month, build, tag)`:

1. Strip an optional leading `v`.
2. Split on `-` into `core` and optional `tag`.
3. Parse `core` as `YYYY.MM.BUILD` (three dot-separated integers).
4. Compare `(year, month, build, tag)` lexicographically; `tag=None` sorts **after** a tagged pre-release of the same core (stable > pre-release).

### Version-aware compatibility

Validation and decode branch on the sidecar's `schema_version`. `vNEXT` requirements are never applied retroactively; a sidecar with `schema_version < vNEXT` is validated and decoded with the legacy meaning below.

| Behaviour | `schema_version < vNEXT` (legacy) | `schema_version >= vNEXT` |
|-----------|-----------------------------------|---------------------------|
| `decode.selection` | Not authored; derived: `requires_nms: true` → `nms`, otherwise `none`. | Required. `requires_nms`, when present, must agree. |
| `decode.nms` absent with host NMS | Existing host NMS behaviour (implementation-defined). | Invalid; `nms` required for `selection: nms`. |
| `decode.max_detections` absent | Default `300`. | No default; valid only with `selection: none`. |
| `decode.max_detections` on a bottom-up sidecar | Also the fixed instance count of the `poses` tensor. | Detection row count only; the instance count is the row count. |
| `may_include_non_person_classes` | Accepted; `is_person_only = not may_include_non_person_classes` (default `true` → `false`). | Invalid; use `is_person_only`. |
| `letterbox` + `preserve_aspect_ratio: false` | Alias for `method: stretch`. | Invalid; use `stretch`. |
| Letterbox `pad_position` / `rounding` absent | Existing host behaviour (implementation-defined). | Required. |
| `color_format` absent with `none`/`unit_float` | Existing host channel order. | Required. |
| `clamp_boxes` absent | Existing host behaviour. | Default `true`. |
| `normalization: imagenet_bgr` | The RGB-ordered tuple — mean `(123.675, 116.28, 103.53)`, std `(58.395, 57.12, 57.375)` — applied in order to the B, G, R channels of the BGR frame (B uses `123.675`). This is the rtmlib convention used by the in-repo RTMPose and RTMW sidecars. | BGR-ordered statistics (B uses `103.53`); see [Normalization modes](#normalization-modes). To keep the legacy behaviour in a `vNEXT` sidecar, declare `normalization: {mode: custom, color_format: bgr, mean: [123.675, 116.28, 103.53], std: [58.395, 57.12, 57.375]}`. |
| Size-level keys other than `input.shape`, `input.resize.target_size`, `onnx` | Deep-merged over the top level. | Invalid (family invariant). |
| `boxes`/`scores`/`class_ids`/`class_logits` semantics, `packed_keypoints`, `stretch`, `class_id_map`, output dtypes `int32`/`int64`/`bool` | Not available. | Available. |

## Validation rules

A sidecar is valid when all of the following hold. Rules marked **(vNEXT)** apply only when `schema_version >= vNEXT`; legacy sidecars follow [Version-aware compatibility](#version-aware-compatibility).

**Identity and composition**

- [ ] `schema_version` is a string matching the skellytracker version pattern (no pre-release tag).
- [ ] `schema_version` is supported by the installed skellytracker release (`installed >= schema_version`).
- [ ] `model_id` matches the sidecar basename (`{model_id}.yaml`).
- [ ] `role` is a non-empty subset of `object_detector`/`pose_estimator`.
- [ ] `display_name`, `input` (with `name`), and a non-empty `outputs` array (each with `name`) are present.
- [ ] `$ref` directives resolve to existing files, contain only the `$ref` key, and terminate (no cycles).
- [ ] `base` directives resolve and merge (with `null`-delete), and the derived file overrides `model_id` and `schema_version`.

**Sizes and artifacts**

- [ ] `sizes` is present with at least one size.
- [ ] Each size declares `input.shape` with `-1` as its batch axis (first dimension).
- [ ] **(vNEXT)** A size entry contains only `input.shape`, `input.resize.target_size`, and `onnx`.
- [ ] Each size's `input.shape` spatial dims equal its `resize.target_size` when `method` is `letterbox`, `stretch`, or `affine_person_crop` and `supports_dynamic_size` is not set. `target_size` is omitted when `method: none` or `supports_dynamic_size: true`; spatial dims are `-1` when `supports_dynamic_size: true`.
- [ ] `input.resize.target_size` (when present) is a length-2 `[height, width]` array.
- [ ] Each size's `onnx.batch_artifacts` has quoted-string positive integer batch keys (`"1"`, `"2"`, …) or a single `dynamic` key (never mixed), and non-empty per-group `precision_artifacts`.
- [ ] Precision keys are the closed enum `fp32`/`fp16`/`int8`.
- [ ] Each precision artifact declares `filename`; when it declares `url`, it also declares `url_sha256`.
- [ ] A precision artifact's `input_dtype` (when authored) agrees with `input.dtype[precision]`.
- [ ] When a size has `len(batch_artifacts) > 1`, each group declares `output_shapes`, whose length equals `len(outputs)` (parallel to `outputs` order) and whose entries are arrays.
- [ ] `batching.batch_axis` (when present) is `0`.

**Input, resize, normalization**

- [ ] `input.dtype` is present; its keys are a subset of `fp32`/`fp16`/`int8` covering the precisions declared in `batch_artifacts`, and its values are `float32`, `float16`, `uint8`, or `int8`. `input.layout` (when present) is `NCHW` or `NHWC`.
- [ ] `input.resize.method` is `letterbox`, `stretch`, `affine_person_crop`, or `none` (`stretch` **(vNEXT)**), and `input.resize.interpolation` is `linear`, `area`, `cubic`, or `nearest`.
- [ ] **(vNEXT)** A `letterbox` resize declares `pad_position` (`top_left` | `center`) and `rounding` (`half_even` | `half_up`); `preserve_aspect_ratio`, when present, is `true`.
- [ ] `pad_value`, `pad_position`, `rounding`, and `preserve_aspect_ratio` are present only when `method: letterbox`; `crop_policy` only when `method: affine_person_crop`.
- [ ] `pad_value` and `crop_policy.pad_value` (when present) are a number or a length-3 array of numbers.
- [ ] `input.resize.crop_policy.expand_ratio` (when present) is a positive float, and `crop_policy.maintain_aspect_ratio` (when present) is a boolean.
- [ ] `input.normalization` (when present) is a valid mode (`none`, `unit_float`, `imagenet_bgr`, `imagenet_rgb`, or `custom` object). A `custom` object has `mode: custom`, length-3 `mean`/`std` when present, a positive `scale` when present, and `color_format` `rgb`/`bgr` when present.
- [ ] `normalization_by_precision` keys are a subset of `fp32`/`fp16`/`int8`, and its values are valid mode strings — never a `custom` object.
- [ ] **(vNEXT)** `input.color_format` is present when any declared precision resolves to `none` or `unit_float`; when present it is `rgb`/`bgr` and agrees with `imagenet_*` and `custom.color_format`.

**Outputs (common)**

- [ ] `outputs[].name` values are unique.
- [ ] Each output's `dtype` is present; its keys are a subset of `fp32`/`fp16`/`int8` covering the precisions declared in `batch_artifacts`, and its values are `float32`, `float16`, `uint8`, `int8`, `int32`, `int64`, or `bool` (the last three **(vNEXT)**).
- [ ] On each output, `rank` (when present) equals `len(shape)`; `rank` is omitted when `shape` is omitted.
- [ ] Each output's `semantic` is one of `detections`, `boxes`, `scores`, `class_ids`, `class_logits`, `simcc_x`, `simcc_y`, `simcc_z`, `heatmap`, `keypoints`, `poses` (`boxes`/`scores`/`class_ids`/`class_logits` **(vNEXT)**). The detection semantics require `role` to include `object_detector`; the pose semantics require `role` to include `pose_estimator`.
- [ ] All sizes share the same output names, semantics, and keypoint/class counts (spatial-dependent dims may scale with input size).

**Detection (`object_detector`)**

- [ ] The sidecar declares `decode` and exactly one primary detection output (`detections` or `boxes`); at most one each of `scores`, `class_ids`, `class_logits`; `class_ids` and `class_logits` are not both present.
- [ ] A `detections` output declares `fields`; `len(fields) == shape[-1]` when `shape[-1]` is concrete and `packed_keypoints` is absent, and `len(fields) == packed_keypoints.offset` otherwise. `fields` is absent on every other semantic.
- [ ] A `boxes` output's last dimension is `4`. `scores`/`class_ids` shapes equal the primary shape without its last dimension. `class_logits` non-class axes equal the `boxes` outer axes, `class_axis` (when present) is a valid non-batch axis, and the class dimension is concrete.
- [ ] `class_ids` output dtypes are `int32`/`int64`; `scores`, `class_logits`, `boxes`, and `detections` dtypes are float types.
- [ ] `decode.boxes_output`, `scores_output`, `class_ids_output`, and `class_logits_output` (when present) name an output with the matching semantic; `boxes_output` names the primary output.
- [ ] `decode.score_field` / `decode.class_field` (when present) appear in the primary `detections` output's `fields`, and are absent when the primary output is `boxes`. `score_field` and `scores_output` are not both present; `class_field`, `class_ids_output`, and `class_logits_output` are mutually exclusive.
- [ ] A `detections` primary output has a score source (`score_field` or `scores_output`) unless `selection: top_k`; a `boxes` primary output has `scores_output` or `class_logits_output`.
- [ ] **(vNEXT)** `decode.selection` is present and is `none`, `top_k`, or `nms`; `requires_nms` (when present) is `true` iff `selection: nms`; `may_include_non_person_classes` is absent.
- [ ] `selection: top_k` requires a `boxes` primary output, `class_logits_output`, and a positive integer `top_k`, and forbids `nms`, `max_detections`, `scores_output`, and `score_field`.
- [ ] `selection: nms` requires `nms` with `iou_threshold` and `score_threshold` (floats in `[0, 1]`), forbids `class_logits_output`, `top_k`, and `max_detections`; `nms.class_agnostic` (when present) is a boolean; `nms.post_nms_top_k` (when present) is a positive integer.
- [ ] `selection: none` forbids `top_k`, `nms`, and `class_logits_output`. `max_detections` (when present) is a positive integer and equals the primary output's row dimension when that is concrete.
- [ ] `decode.box_format` (when present) is `xyxy`, `xywh`, or `cxcywh`; `decode.box_scale` (when present) is `pixel`, `normalized_01`, or `normalized_11`; `decode.score_activation` (when present) is `none` or `sigmoid`; `decode.clamp_boxes` and `decode.is_person_only` (when present) are booleans.
- [ ] `decode.class_id_map` (when present) has length `C` (the class dimension of `class_logits`, or the model's class count), values are `-1` or non-negative integers, and non-negative values are unique.
- [ ] `decode.class_id_base` and `decode.person_class_id` are non-negative integers. With `class_id_map`, `person_class_id` is one of its non-negative values; otherwise `person_class_id >= class_id_base`.
- [ ] `decode.confidence_threshold_default` (when present) is in `[0, 1]`.

**Pose (`pose_estimator`)**

- [ ] The sidecar declares `pose` (with `estimator_type` and `requires_object_detector`), `pose.decode`, `pose.tracked_points`, and `pose.connections`.
- [ ] `pose.requires_object_detector` is a boolean and consistent with `pose.estimator_type`: `true` for `top_down_single_person`/`top_down_multi_person`, `false` for `bottom_up`.
- [ ] `pose.estimator_type` is one of `top_down_single_person`, `top_down_multi_person`, `bottom_up`; `pose.decode.method` is one of `simcc`, `heatmap`, `coordinate`.
- [ ] A sidecar with `pose.requires_object_detector: true` declares `input.resize.method: affine_person_crop`.
- [ ] `pose.tracked_points` names are unique.
- [ ] Every `pose.connections[].edges` endpoint appears in `pose.tracked_points` or `pose.derived_points`, and `pose.connections[].name` values are unique.
- [ ] `pose.derived_points` names are unique; each derivation is a string (passthrough), a list of names (mean), or a name→number map (weighted), and every referenced source name appears in `pose.tracked_points`.
- [ ] `pose.canonical_mapping` (when present) has unique keys; each value is a string (passthrough), a list of names (mean), or a name→number map (weighted), and every referenced source name appears in `pose.tracked_points`.
- [ ] A `simcc` decoder declares exactly one `simcc_x` and one `simcc_y` output, plus one `simcc_z` output iff `is_3d: true`; `heatmap` declares exactly one output with `semantic: heatmap`; `coordinate` declares exactly one `keypoints` output (single-person), or — for `bottom_up` — one primary detection output plus either one `poses` output or `packed_keypoints` on the `detections` output (not both).
- [ ] A `poses` output declares `keypoint_axis: 2`, and its instance axis equals the primary detection output's row axis.
- [ ] `packed_keypoints` (when present) is on a `detections` output of a sidecar with both roles, `estimator_type: bottom_up`, and `pose.decode.method: coordinate`; `offset` equals `len(fields)`; `components` is a list of unique values from `x`, `y`, `z`, `score` containing `x` and `y`, and containing `z` iff `is_3d: true`; `offset + len(tracked_points) * len(components)` equals `shape[-1]` when concrete; and `pose.decode.coordinate_scale` (default `pixel`) equals `decode.box_scale` (default `pixel`).
- [ ] `keypoint_count` (when authored on a pose output) equals `len(pose.tracked_points)`.
- [ ] `pose.decode.is_3d` (when present) is a boolean; when `true`, `pose.decode.depth_unit` is present and one of `m`, `mm`, `pixel`; `depth_unit` and `depth_range` are present only when `is_3d: true`.
- [ ] When `pose.decode.is_3d: true` and `pose.decode.method: simcc`, `pose.decode.depth_range` is present and is a length-2 array `[min, max]` in `depth_unit`.
- [ ] `pose.decode.method: heatmap` with `pose.decode.is_3d: true` is invalid (3D heatmap is not supported).
- [ ] `pose.decode.coordinate_scale` is present only when `pose.decode.method: coordinate`, and (when present) is `pixel`, `normalized_01`, or `normalized_11`.

**Overlay**

- [ ] `overlay.skeleton` (when a string) names an entry in `pose.connections`; when an inline edge list, every endpoint appears in `pose.tracked_points` or `pose.derived_points`.
- [ ] Each `overlay.groups.<name>` declares `connection_color` and uses `connections` or `prefix` (not both); at most one group has neither (the default group).
- [ ] `overlay.groups.<name>.connection_thickness` (when present) is a positive integer.
- [ ] Every edge in a group's `connections` is a member of the selected skeleton's edge set (the named `pose.connections[]` entry, or the inline `overlay.skeleton` edge list).
- [ ] `overlay.keypoint_color`, `overlay.groups.<name>.connection_color`, and `overlay.groups.<name>.keypoint_color` (when present) are length-3 arrays.

Every future contract change must bump `schema_version` to the skellytracker release that ships it and add a changelog row below.

## Changelog

| `schema_version` | Changes |
|------------------|---------|
| `v2024.09.1019` | Initial YAML sidecar spec: one `{model_id}.yaml` per family with a `sizes` map (size × batch × precision via `sizes.<size>.onnx.batch_artifacts`, incl. `dynamic` batch); `role` arrays (`object_detector` / `pose_estimator`, both for one-stage models); `input.normalization` modes (`none`, `unit_float`, `imagenet_bgr`, `imagenet_rgb`, `custom`) with per-precision overrides; `input.resize` (`letterbox` + `affine_person_crop` with `crop_policy`, `none`, interpolation); file composition (`$ref` includes + `base` inheritance with `null`-delete); `schema_version` as skellytracker version string; pose output contract (`tracked_points`, named `connections`, `derived_points`, `canonical_mapping` with `prefixes`, `overlay.groups`, `requires_object_detector`, and `decode` incl. `simcc`/`heatmap`/`coordinate` with 3D via `is_3d`/`depth_unit`/`depth_range`); VRM 1.0 canonical skeleton with current→VRM name-migration note; `detectors/` storage layout; per-precision `url` with mandatory `url_sha256`. |
| `vNEXT` | Added `input.resize.method: stretch`; letterbox `pad_position` and `rounding` (required), normative letterbox/centre-offset algorithm, per-method unprojection, non-antialiased resizes, scalar/array `pad_value`; `affine_person_crop` warp definition with `crop_policy.pad_value`. Added `input.color_format` (required for `none`/`unit_float`); `none` passes pixels in `input.dtype`; explicit `imagenet_*` channel order (`imagenet_bgr` now uses BGR-ordered statistics; legacy meaning retained for older sidecars). Added detection semantics `boxes`, `scores`, `class_ids`, `class_logits` (with `class_axis`) and output dtypes `int32`/`int64`/`bool`; output names must be unique. Rebuilt detection `decode` around required `selection` (`none`/`top_k`/`nms`) with output bindings, `box_scale`, `score_activation`, `class_id_map`, `top_k`, `nms` (`iou_threshold`, `score_threshold`, `class_agnostic`, `post_nms_top_k`), `clamp_boxes`, `is_person_only`; normative stage order and strict thresholds. `max_detections` has no default and only declares a graph-final row count; `requires_nms` and `may_include_non_person_classes` are legacy. Added `packed_keypoints` for bottom-up packed rows; bottom-up pose rows follow surviving detection rows. Size entries limited to `input.shape`, `input.resize.target_size`, `onnx` (family invariant). Added version-aware compatibility rules. |

## Reference examples

Each example is the sidecar the named exporter writes once it is updated to this revision: tensor names, shapes, dtypes, sizes, batches, precisions and filenames are taken from that exporter's code and published artifacts. Where a family has more sizes than shown, the omitted sizes differ only in `input.shape`, `target_size` and filenames.

### YOLO26 detector (Ultralytics-Exporter; NMS-free, graph-final rows)

Centre letterbox with half-even rounding, RGB `unit_float` input, and one packed `[B, 300, 6]` tensor of final detections (`nms=False`, `max_det=300` at export). Every precision, including int8, keeps float32 I/O.

```yaml
---
schema_version: "vNEXT"
model_id: yolo26
display_name: YOLO26
role: [object_detector]

input:
  name: images
  dtype: {fp32: float32, fp16: float32, int8: float32}
  layout: NCHW
  color_format: rgb
  normalization: unit_float
  resize: {method: letterbox, pad_position: center, rounding: half_even, pad_value: 114, interpolation: linear}

batching: {batch_axis: 0}

outputs:
  - name: output0
    dtype: {fp32: float32, fp16: float32, int8: float32}
    shape: [-1, 300, 6]
    rank: 3
    semantic: detections
    fields: [x1, y1, x2, y2, score, class_id]

decode:
  selection: none
  max_detections: 300
  box_format: xyxy
  box_scale: pixel
  score_field: score
  class_field: class_id
  class_id_base: 0
  person_class_id: 0
  is_person_only: false
  confidence_threshold_default: 0.7

sizes:
  nano:
    input:
      shape: [-1, 3, 640, 640]
      resize:
        target_size: [640, 640]
    onnx:
      batch_artifacts:
        "1":
          precision_artifacts:
            fp32: {filename: yolo26-nano_b1_fp32.onnx}
            fp16: {filename: yolo26-nano_b1_fp16.onnx}
            int8: {filename: yolo26-nano_b1_int8.onnx}
          output_shapes:
            - [1, 300, 6]
        "2":
          precision_artifacts:
            fp32: {filename: yolo26-nano_b2_fp32.onnx}
            fp16: {filename: yolo26-nano_b2_fp16.onnx}
            int8: {filename: yolo26-nano_b2_int8.onnx}
          output_shapes:
            - [2, 300, 6]
        "3":
          precision_artifacts:
            fp32: {filename: yolo26-nano_b3_fp32.onnx}
            fp16: {filename: yolo26-nano_b3_fp16.onnx}
            int8: {filename: yolo26-nano_b3_int8.onnx}
          output_shapes:
            - [3, 300, 6]
  # small, medium, large, xlarge: identical to nano (640x640, batches 1-3) except filenames.
```

The input shape for a group is derived: `[N] + input.shape[1:]` (batch key `N` + the size's image shape). When a size lists only one batch key, the top-level `outputs[].shape` may be used instead of per-group `output_shapes`.

### YOLO26 pose (Ultralytics-Exporter; bottom-up, packed keypoints)

The pose export emits one `[B, 300, 57]` tensor: 6 detection columns followed by 17 × `(x, y, score)`, with keypoints in model-input pixels and keypoint scores already sigmoid-activated. It is declared once, with `packed_keypoints`.

```yaml
---
schema_version: "vNEXT"
model_id: yolo26-body
display_name: YOLO26 Pose
role: [object_detector, pose_estimator]

input:
  name: images
  dtype: {fp32: float32, fp16: float32, int8: float32}
  layout: NCHW
  color_format: rgb
  normalization: unit_float
  resize: {method: letterbox, pad_position: center, rounding: half_even, pad_value: 114, interpolation: linear}

batching: {batch_axis: 0}

outputs:
  - name: output0
    dtype: {fp32: float32, fp16: float32, int8: float32}
    shape: [-1, 300, 57]
    rank: 3
    semantic: detections
    fields: [x1, y1, x2, y2, score, class_id]
    packed_keypoints:
      offset: 6
      components: [x, y, score]

decode:
  selection: none
  max_detections: 300
  box_format: xyxy
  box_scale: pixel
  score_field: score
  class_field: class_id
  class_id_base: 0
  person_class_id: 0
  is_person_only: true
  confidence_threshold_default: 0.3

pose:
  estimator_type: bottom_up
  requires_object_detector: false
  landmark_schema: COCO17
  tracked_points: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow,
    left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]
  connections:
    - name: coco_17
      edges:
        - [nose, left_eye]
        - [nose, right_eye]
        - [left_eye, left_ear]
        - [right_eye, right_ear]
        - [left_shoulder, right_shoulder]
        - [left_shoulder, left_elbow]
        - [left_elbow, left_wrist]
        - [right_shoulder, right_elbow]
        - [right_elbow, right_wrist]
        - [left_shoulder, left_hip]
        - [right_shoulder, right_hip]
        - [left_hip, right_hip]
        - [left_hip, left_knee]
        - [left_knee, left_ankle]
        - [right_hip, right_knee]
        - [right_knee, right_ankle]
  decode:
    method: coordinate
    coordinate_scale: pixel

overlay:
  skeleton: coco_17
  keypoint_color: [0, 183, 235]
  groups:
    left:
      prefix: left_
      connection_color: [0, 200, 83]
    right:
      prefix: right_
      connection_color: [255, 82, 82]
    body:
      connection_color: [0, 183, 235]

sizes:
  nano:
    input:
      shape: [-1, 3, 640, 640]
      resize:
        target_size: [640, 640]
    onnx:
      batch_artifacts:
        "1":
          precision_artifacts:
            fp32: {filename: yolo26-body-nano_b1_fp32.onnx}
            fp16: {filename: yolo26-body-nano_b1_fp16.onnx}
            int8: {filename: yolo26-body-nano_b1_int8.onnx}
          output_shapes:
            - [1, 300, 57]
        "2":
          precision_artifacts:
            fp32: {filename: yolo26-body-nano_b2_fp32.onnx}
            fp16: {filename: yolo26-body-nano_b2_fp16.onnx}
            int8: {filename: yolo26-body-nano_b2_int8.onnx}
          output_shapes:
            - [2, 300, 57]
        "3":
          precision_artifacts:
            fp32: {filename: yolo26-body-nano_b3_fp32.onnx}
            fp16: {filename: yolo26-body-nano_b3_fp16.onnx}
            int8: {filename: yolo26-body-nano_b3_int8.onnx}
          output_shapes:
            - [3, 300, 57]
  # small, medium, large, xlarge: identical to nano except filenames.
```

### RF-DETR (Roboflow-Exporter; box tensor + class logits, global top-k)

RF-DETR emits normalized `cxcywh` boxes and raw logits over 91 sparse COCO category slots. The host stretches the image, applies ImageNet RGB normalization (the graph contains none), and selects the top 300 (query, class) pairs; there is no NMS. Artifacts are static-batch (1–3), fp32 and fp16, all with float32 I/O.

```yaml
---
schema_version: "vNEXT"
model_id: rfdetr
display_name: RF-DETR
role: [object_detector]

input:
  name: input
  dtype: {fp32: float32, fp16: float32}
  layout: NCHW
  color_format: rgb
  normalization: imagenet_rgb
  resize: {method: stretch, interpolation: linear}

batching: {batch_axis: 0}

outputs:
  - name: dets
    dtype: {fp32: float32, fp16: float32}
    shape: [-1, 300, 4]
    rank: 3
    semantic: boxes
  - name: labels
    dtype: {fp32: float32, fp16: float32}
    shape: [-1, 300, 91]
    rank: 3
    semantic: class_logits
    class_axis: -1

decode:
  selection: top_k
  top_k: 300
  boxes_output: dets
  class_logits_output: labels
  box_format: cxcywh
  box_scale: normalized_01
  score_activation: sigmoid
  class_id_map: {$ref: ../../shared/class_ids/coco91.yaml}
  person_class_id: 1          # COCO category id of person
  is_person_only: false
  confidence_threshold_default: 0.5
  clamp_boxes: true

sizes:
  nano:
    input:
      shape: [-1, 3, 384, 384]
      resize:
        target_size: [384, 384]
    onnx:
      batch_artifacts:
        "1":
          precision_artifacts:
            fp32: {filename: rfdetr-nano_b1_fp32.onnx}
            fp16: {filename: rfdetr-nano_b1_fp16.onnx}
          output_shapes:
            - [1, 300, 4]
            - [1, 300, 91]
        "2":
          precision_artifacts:
            fp32: {filename: rfdetr-nano_b2_fp32.onnx}
            fp16: {filename: rfdetr-nano_b2_fp16.onnx}
          output_shapes:
            - [2, 300, 4]
            - [2, 300, 91]
        "3":
          precision_artifacts:
            fp32: {filename: rfdetr-nano_b3_fp32.onnx}
            fp16: {filename: rfdetr-nano_b3_fp16.onnx}
          output_shapes:
            - [3, 300, 4]
            - [3, 300, 91]
  # small (512), medium (576), large (704): same pattern with their own input size and filenames.
```

`detectors/shared/class_ids/coco91.yaml` is a 91-entry sequence: index = model slot, value = emitted COCO category id, `-1` = ignore. Slot 0 and the ten COCO id gaps (12, 26, 29, 30, 45, 66, 68, 69, 71, 83) are `-1`; every other slot `i` maps to `i`, giving 80 non-negative entries. Excerpt:

```yaml
- -1     # slot 0: unused
- 1      # person
- 2
# ... slots 3-10
- 11
- -1     # slot 12: COCO id gap
- 13
# ... through slot 90
- 90     # toothbrush
```

### RTMDet person (OpenMMLab-Exporter; separate class-id tensor, host NMS)

The mmdet test pipeline letterboxes into the top-left corner with half-up rounding and pad 114, normalizing BGR pixels. The graph emits pre-NMS candidates — `dets` `[B, Q, 5]` and int64 `labels` `[B, Q]`, where `Q` varies per image — and the host runs the family's baseline NMS (`score_thr` 0.05, IoU 0.6, `max_per_img` 100). The graph reports the last two `dets` dims symbolically; the trailing `5` is `len(fields)`.

```yaml
---
schema_version: "vNEXT"
model_id: rtmdet-person
display_name: RTMDet Person
role: [object_detector]

input:
  name: input
  dtype: {fp32: float32, fp16: float32}
  layout: NCHW
  color_format: bgr
  normalization: imagenet_bgr
  resize: {method: letterbox, pad_position: top_left, rounding: half_up, pad_value: 114, interpolation: linear}

batching: {batch_axis: 0}

outputs:
  - name: dets
    dtype: {fp32: float32, fp16: float32}
    shape: [-1, -1, 5]
    rank: 3
    semantic: detections
    fields: [x1, y1, x2, y2, score]
  - name: labels
    dtype: {fp32: int64, fp16: int64}
    shape: [-1, -1]
    rank: 2
    semantic: class_ids

decode:
  selection: nms
  boxes_output: dets
  box_format: xyxy
  box_scale: pixel
  score_field: score
  class_ids_output: labels
  class_id_base: 0
  person_class_id: 0
  is_person_only: true
  confidence_threshold_default: 0.7
  clamp_boxes: true
  nms:
    iou_threshold: 0.6
    score_threshold: 0.05
    class_agnostic: false
    post_nms_top_k: 100

sizes:
  nano:
    input:
      shape: [-1, 3, 320, 320]
      resize:
        target_size: [320, 320]
    onnx:
      batch_artifacts:
        "1":
          precision_artifacts:
            fp32: {filename: rtmdet-person-nano_b1_fp32.onnx}
            fp16: {filename: rtmdet-person-nano_b1_fp16.onnx}
          output_shapes:
            - [1, -1, 5]
            - [1, -1]
        "2":
          precision_artifacts:
            fp32: {filename: rtmdet-person-nano_b2_fp32.onnx}
            fp16: {filename: rtmdet-person-nano_b2_fp16.onnx}
          output_shapes:
            - [2, -1, 5]
            - [2, -1]
        "3":
          precision_artifacts:
            fp32: {filename: rtmdet-person-nano_b3_fp32.onnx}
            fp16: {filename: rtmdet-person-nano_b3_fp16.onnx}
          output_shapes:
            - [3, -1, 5]
            - [3, -1]
  medium:
    input:
      shape: [-1, 3, 640, 640]
      resize:
        target_size: [640, 640]
    onnx:
      batch_artifacts:
        "1":
          precision_artifacts:
            fp32: {filename: rtmdet-person-medium_b1_fp32.onnx}
            fp16: {filename: rtmdet-person-medium_b1_fp16.onnx}
          output_shapes:
            - [1, -1, 5]
            - [1, -1]
        # "2" and "3": same pattern as nano.
```

The CSPNeXt COCO family (`rtmdet`: tiny, small, medium, large, xlarge at 640; x-p6 at 1280) has the same shape with `is_person_only: false` and `nms: {iou_threshold: 0.65, score_threshold: 0.001, class_agnostic: false, post_nms_top_k: 300}`. The ConvNeXt/Swin COCO sizes form a third family with the same decode as `rtmdet` but `color_format: rgb` and `normalization: imagenet_rgb`.

### RTMW WholeBody (OpenMMLab-Exporter; top-down pose)

mmpose `input_size` is `(width, height)`: the `256x192` models take `[N, 3, 256, 192]` input and the `384x288` models `[N, 3, 384, 288]`. The model consumes RGB (`bgr_to_rgb=True`) normalized with ImageNet statistics. SIMCC bin counts are twice the input width/height (`simcc_split_ratio=2.0`) and are read from the output tensors, so `shape` is omitted.

```yaml
---
schema_version: "vNEXT"
model_id: rtmw-wholebody
display_name: RTMW WholeBody
role: [pose_estimator]

input:
  name: input
  dtype: {fp32: float32, fp16: float32}
  layout: NCHW
  color_format: rgb
  normalization: imagenet_rgb
  resize:
    method: affine_person_crop
    interpolation: linear
    crop_policy:
      expand_ratio: 1.25
      maintain_aspect_ratio: true
      pad_value: 0

batching: {batch_axis: 0}

outputs:
  - name: simcc_x
    dtype: {fp32: float32, fp16: float32}
    semantic: simcc_x
    keypoint_axis: 1
    keypoint_count: 133
  - name: simcc_y
    dtype: {fp32: float32, fp16: float32}
    semantic: simcc_y
    keypoint_axis: 1
    keypoint_count: 133

pose:
  estimator_type: top_down_single_person
  requires_object_detector: true
  landmark_schema: COCO_WholeBody
  # The exporter currently inlines these 133 names and edges; the shared fragments are equivalent.
  tracked_points: {$ref: ../../../shared/skeletons/coco133_tracked_points.yaml}
  connections: {$ref: ../../../shared/skeletons/coco133_skeletons.yaml}
  decode:
    method: simcc

overlay:
  skeleton: rtmpose_skeleton
  groups:
    right_hand:
      prefix: right_hand_
      connection_color: [0, 100, 255]
    left_hand:
      prefix: left_hand_
      connection_color: [255, 100, 0]
    face:
      prefix: face_
      connection_color: [200, 0, 200]
    body:
      connection_color: [0, 200, 100]

sizes:
  m:
    input:
      shape: [-1, 3, 256, 192]
      resize:
        target_size: [256, 192]
    onnx:
      batch_artifacts:
        "1":
          precision_artifacts:
            fp32: {filename: rtmw-wholebody-m_b1_fp32.onnx}
            fp16: {filename: rtmw-wholebody-m_b1_fp16.onnx}
  l-384:
    input:
      shape: [-1, 3, 384, 288]
      resize:
        target_size: [384, 288]
    onnx:
      batch_artifacts:
        "1":
          precision_artifacts:
            fp32: {filename: rtmw-wholebody-l-384_b1_fp32.onnx}
            fp16: {filename: rtmw-wholebody-l-384_b1_fp16.onnx}
  # l-256, x-256 (256x192) and x-384 (384x288): same pattern.
```

For comparison, skellytracker's in-repo `rtmw/wholebody/rtmw-wholebody.yaml` describes the rtmlib `onnx_sdk` downloads (with `url`/`url_sha256`) at `schema_version: v2024.09.1019`. It uses the legacy `imagenet_bgr` meaning (see [Version-aware compatibility](#version-aware-compatibility)).
