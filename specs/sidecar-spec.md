# Model Sidecar Specification

## Overview

A **sidecar** is a YAML file that sits beside one or more ONNX model files and describes the runtime I/O contract a host must satisfy to load and run them correctly: input tensor, normalization, resize, batching, output semantics, and decode.

Sidecars support two model **roles**:

| Role | Purpose |
|------|---------|
| `detector` | Object detection (e.g. YOLO26) — emits boxes, scores, and class IDs. |
| `pose_estimator` | Pose estimation (e.g. RTMW, RTMPose) — emits keypoint coordinates. |

A sidecar may declare **both** roles — e.g. a one-stage model such as RTMO that emits boxes and keypoints in the same pass. In that case `role` lists both values, and `outputs` mixes `detections` and pose tensors.

### Core properties

- **One sidecar per model family.** A single `{model_id}.yaml` file lists every size of the model, and within each size every native batch size and precision variant, via `sizes.<size>.onnx.batch_artifacts`.
- **Multiple sizes per sidecar.** A family ships several sizes (e.g. `nano`, `small`, `medium`) in one file. All sizes share the output contract (`outputs`, `decode`, `normalization`) and differ only in input size and ONNX artifacts.
- **Traceable compatibility.** A sidecar's `schema_version` identifies the minimum skellytracker release required to consume it.
- **Single source of truth.** This document is the canonical, human-readable contract. It is **not parsed at runtime**; skellytracker mirrors it with Pydantic `BaseModel` subclasses (the `SidecarModel` hierarchy), and documentation and validators must stay aligned.
- **Composable.** A sidecar may `$ref` shared YAML fragments and may `base` another sidecar to inherit and override values.

### Document map

| Section | Purpose |
|---------|---------|
| [Format](#format) | File naming, style conventions, storage layout, and file-pairing rules. |
| [File composition](#file-composition-ref-includes-and-base-inheritance) | `$ref` includes and `base` inheritance for reuse. |
| [Common fields](#common-fields) | Shared identity, input, batching, and output fields for every sidecar. |
| [Detection fields](#detection-fields-detector-role) | Output fields and detection `decode` for the `detector` role. |
| [Pose-estimation fields](#pose-estimation-fields-pose_estimator-role) | Keypoints, skeletons, canonical mapping, and `decode` for the `pose_estimator` role. |
| [Model sizes within a family](#model-sizes-within-a-family) | How multiple sizes share one output contract. |
| [Normalization modes](#normalization-modes) | Pixel normalization options. |
| [Schema versioning](#schema-versioning) | How `schema_version` gates compatibility. |
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
      {family}/                     # role: detector
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
```

| Branch | Role | Contents |
|--------|------|----------|
| `detectors/calibration_detectors/` | calibration | Calibration detectors (e.g. Charuco boards). Hard-coded in skellytracker — **not** specified via sidecars. |
| `detectors/object_detectors/{family}/` | `detector` | One sidecar per family, e.g. `yolo26/`. |
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

# detectors/keypoint_detectors/rtmw/wholebody/rtmw-wholebody.yaml
---
schema_version: "v2024.09.1019"
model_id: rtmw-wholebody
display_name: RTMW WholeBody
role: [pose_estimator]
pose:
  estimator_type: top_down_single_person
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
# detectors/object_detectors/yolo26-int8/yolo26-int8.yaml — int8 variant of the whole family
---
schema_version: "v2024.09.1019"
base: ../yolo26/yolo26.yaml
model_id: yolo26-int8
display_name: YOLO26 (int8)
sizes:
  nano:
    onnx:
      batch_artifacts:
        2:
          precision_artifacts:
            int8:
              filename: yolo26-int8-nano_b2_int8.onnx
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
        2:
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
| `role` | array[enum] | yes | One or both of `detector` / `pose_estimator`, e.g. `[detector]`, `[pose_estimator]`, or `[detector, pose_estimator]` for a one-stage model such as RTMO. |
| `sizes` | object | yes | Map of size name → size definition (see [Sizes](#sizes)). At least one size. |

### `sizes.<size>.onnx.batch_artifacts`

A mapping from **native batch size** → artifact group, declared **per size** under `sizes.<size>.onnx.batch_artifacts`. Keys are either positive integer batch sizes or the literal `dynamic`, which means the model accepts any runtime batch size. Each group contains a `precision_artifacts` map using the per-precision schema below.

The full input tensor shape for a group is `[N] + sizes.<size>.input.shape[1:]` — the batch key `N` supplies the batch dim, and the size's `input.shape` supplies the channels and spatial dims (identical across all precisions and batches for that size).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sizes.<size>.onnx.batch_artifacts` | object | yes per size | Keys are positive integer native batch sizes or `dynamic`. A size may list any number of batch keys. |
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

- Each size's `batch_artifacts` must have at least one key; each key must be a positive integer or `dynamic`.
- A size's `batch_artifacts` may contain integer keys **or** a single `dynamic` key — never both. Mixing `dynamic` with integer keys is a validation error.
- Each group must list at least one precision.
- Precision keys are the closed enum `fp32`, `fp16`, `int8`.
- `batching.native_batch_sizes` is **derived** per size from the integer keys in `batch_artifacts`. A `dynamic` key means any runtime batch size is accepted — it is not a separate field.
- A precision artifact's `input_dtype` (when authored) must agree with `input.dtype` for that precision.

### `input`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | ONNX input tensor name. |
| `dtype` | object | yes | Map of precision → ONNX graph input element type, e.g. `{fp32: float32, fp16: float16, int8: uint8}`. When all variants share one type, every entry is that type (e.g. `{fp32: float32, fp16: float32, int8: float32}`). Keys are a subset of `fp32`/`fp16`/`int8` covering the precisions declared in `batch_artifacts`. |
| `layout` | enum | no | `NCHW` (default) or `NHWC`. |
| `normalization` | string or object | no | See [Normalization modes](#normalization-modes). Default `imagenet_bgr` (the RTMPose convention). |
| `normalization_by_precision` | object | no | Optional `fp32`/`fp16`/`int8` → named mode override. Values are mode strings only (no `custom` object); a `custom` normalization belongs at the top-level `normalization`. |
| `resize` | object | no | Resize config (see below); omitted only when the model consumes the source image unchanged (same effect as `method: none`). `target_size` is per-size (`sizes.<size>.input.resize.target_size`); other fields are shared. |

`shape` is **not** a top-level field — the input image shape is declared per size as `sizes.<size>.input.shape` (see [Sizes](#sizes)). Its batch axis is always `-1`: the concrete native batch size lives in the `batch_artifacts` keys, never in the shape. The channels and spatial dims are concrete, except that the spatial dims are `-1` when [`input.resize.supports_dynamic_size`](#inputresize) is `true`.

`input` is required for `detector` and `pose_estimator` sidecars. This spec models **single-input** models only — `input` describes exactly one input tensor; multi-input models are out of scope.

Decoded box and keypoint coordinates are always in a **top-left origin** — `x` grows right, `y` grows down — in the model's input image; the host unprojects them to the source image. There is no per-sidecar origin override.

### `input.resize`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `method` | enum | yes | `letterbox` (detector default), `affine_person_crop` (top-down pose), or `none` (pass the source image unchanged — no resize). |
| `target_size` | array[2] | conditional | `[height, width]` the input is resized to. Declared in `sizes.<size>.input.resize.target_size`. Required when `resize` is present and `method` is `letterbox` or `affine_person_crop`, unless `supports_dynamic_size: true`. Omitted when `method: none`. |
| `supports_dynamic_size` | bool | no | Model accepts any input spatial size; `target_size` is omitted. Default `false`. |
| `preserve_aspect_ratio` | bool | no | Letterbox only; default `true`. |
| `pad_value` | int | no | Letterbox only; default `114`. |
| `interpolation` | enum | no | Host resize filter; default `linear`. |
| `crop_policy` | object | no | `affine_person_crop` only; see below. |

When `supports_dynamic_size: true`, the model accepts any input spatial size: `target_size` is omitted and the size's `input.shape` spatial dims are `-1`. This applies to both `letterbox` (the host pads to the runtime-accepted size) and `affine_person_crop` (the host warps the crop to the runtime-accepted size). When it is not set and `method` is `letterbox` or `affine_person_crop`, `target_size` is required.

**`none`**

The model consumes the source image directly — no letterbox, no affine crop. `target_size` is omitted and the host passes the source image unchanged. The size's `input.shape` spatial dims are concrete and declare the exact resolution the model expects (the caller supplies frames at that size), unless `supports_dynamic_size: true`, in which case any spatial size is accepted.

**`affine_person_crop`**

The top-down pose convention: a person bounding box is warped to `target_size` by an affine transform (center + scale) rather than padding a full frame. Used by top-down models such as RTMW and RTMW3D. `crop_policy` controls how the source bounding box is expanded before the warp:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input.resize.crop_policy.expand_ratio` | float | no | BBox expansion factor before the affine warp; default `1.25`. |
| `input.resize.crop_policy.maintain_aspect_ratio` | bool | no | Default `true`. |

**Interpolation values:**

| Value | OpenCV constant | Typical use |
|-------|-----------------|-------------|
| `linear` | `cv2.INTER_LINEAR` | YOLO/Ultralytics letterbox export (default) |
| `area` | `cv2.INTER_AREA` | Downscaling with area resampling |
| `cubic` | `cv2.INTER_CUBIC` | Higher-quality upscale |
| `nearest` | `cv2.INTER_NEAREST` | Nearest-neighbor |

### `sizes`

The top-level `sizes` map enumerates every scale of the model in a single sidecar.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sizes.<name>.input` | object | yes per size | Size-specific input: `shape` (required; batch axis always `-1`) and `resize.target_size` (required when `resize` is present, `method` is `letterbox`/`affine_person_crop`, and `supports_dynamic_size` is not set). |
| `sizes.<name>.onnx` | object | yes per size | Size-specific artifacts: `batch_artifacts` (required). |

A size entry is a **partial document deep-merged over the top level** (excluding `sizes` itself) using the same merge semantics as [`base`](#base--inherit-from-another-sidecar-and-override). Size names are lowercase tokens such as `nano`, `small`, `medium`; hyphens are allowed (e.g. `l-m`, `x-l`).

When `target_size` is declared, the size's `input.shape` spatial dimensions must equal it (for `NCHW`, `shape[2:4] == target_size`; for `NHWC`, `shape[1:3] == target_size`); with `supports_dynamic_size: true`, `target_size` is omitted. The first size listed is the default when no size is requested. How a host selects a size or precision at runtime is otherwise out of scope — this spec only declares what is available.

| Contract part | Where it lives |
|---------------|----------------|
| `role`, `outputs`, `input` (except `shape` / `resize.target_size`), `batching` | shared at top level |
| `decode` | top level, when `role` includes `detector` |
| `pose`, `overlay` | top level, when `role` includes `pose_estimator` |
| `input.shape`, `input.resize.target_size` (unless `supports_dynamic_size`), `onnx.batch_artifacts` | per size (`sizes.<size>`) |

### `batching`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `batch_axis` | int | no | Batch dimension axis; default `0`. Non-zero batch axes are out of scope. |
| `supports_dynamic_batch` | bool | derived | `true` when `batch_artifacts` contains a `dynamic` key. Derived at load time — not authored. |

> `batching.batch_conversion` is **not part of this spec** and is reserved for a future change set.

### `outputs` (common)

Each entry describes one ONNX output tensor. `outputs` is required for every sidecar. The fields below are common to every role; role-specific fields are described in the [detection](#detection-fields-detector-role) and [pose-estimation](#pose-estimation-fields-pose_estimator-role) sections.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | ONNX output tensor name. |
| `dtype` | object | yes | Map of precision → output element type, like `input.dtype` (e.g. `{fp32: float32, fp16: float16, int8: uint8}`). May differ from the input precision — e.g. an fp16/int8 model whose outputs are emitted as float32. |
| `shape` | array | no | Output tensor shape. Omitted for pose outputs whose shape varies with the input size (e.g. SIMCC bins). |
| `rank` | int | no | Tensor rank. Omitted when `shape` is omitted. |
| `semantic` | enum | yes | Closed set: `detections`, `simcc_x`, `simcc_y`, `simcc_z`, `heatmap`, `keypoints`, `poses`. Links the tensor to `decode` / `pose.decode`. |

All sizes in a sidecar share one output contract: each output tensor has the same semantics and keypoint count across sizes. Fixed-shape outputs (e.g. `detections`) differ only in the batch dimension; spatial-dependent pose outputs (SIMCC bins, heatmap H×W) may scale their spatial dims with the input size. Per-batch shapes are declared via `batch_artifacts.<N>.output_shapes`.

A sidecar with both roles lists every tensor in one `outputs` array. Each tensor's `semantic` routes it to its decoder: `detections` → `decode` (detection); pose semantics → `pose.decode`. Semantics map to roles as follows: `detections` requires `role` to include `detector` (a both-role model such as RTMO emits it as well); `simcc_x`/`simcc_y`/`simcc_z`/`heatmap`/`keypoints`/`poses` require `role` to include `pose_estimator`.

```yaml
# One-stage detector + pose model (role: [detector, pose_estimator])
outputs:
  - name: out_det
    dtype:
      fp32: float32
    shape: [1, 300, 5]
    rank: 3
    semantic: detections
    fields: [x1, y1, x2, y2, score]
  - name: out_pose
    dtype:
      fp32: float32
    shape: [1, 300, 17, 3]
    rank: 4
    semantic: poses
    keypoint_axis: 2

decode:                # detection decode — applies to the `detections` tensor
  box_format: xyxy
  score_field: score
  requires_nms: true

pose:                  # pose decode — applies to the `poses` tensor
  estimator_type: bottom_up
  requires_detector: false
  # ... tracked_points / connections ...
  decode:
    method: coordinate
```

## Detection fields (`detector` role)

Fields that apply when `role` includes `detector`.

### `outputs` (detection)

Additional fields on `detector` output tensors — the raw tensor schema:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `fields` | array | conditional | Ordered field names per detection row (e.g. `[x1, y1, x2, y2, score, class_id]`). Required when `semantic: detections`. |

### `decode` (detection)

Required when `role` includes `detector`. How the host interprets the output whose `semantic` is `detections` — box format, score/class fields, and post-processing defaults. Box coordinates are expressed in the model's input image (post-resize) scale; the host unprojects them to the source image:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `box_format` | enum | no | `xyxy`, `xywh`, or `cxcywh`; the first four entries of `fields` are the box coordinates in this format. |
| `score_field` | string | no | Field name for confidence (e.g. `score`); must appear in `fields`. |
| `class_field` | string | no | Field name for class ID (e.g. `class_id`); must appear in `fields`. |
| `class_id_base` | int | no | Lowest class ID value in the model's output; default `0`. |
| `person_class_id` | int | no | Default `0`. |
| `max_detections` | int | no | Default `300`. Also the fixed instance count (`N`) of a bottom-up `poses` tensor. |
| `may_include_non_person_classes` | bool | no | Default `true`. |
| `requires_nms` | bool | no | Default `false`. |
| `confidence_threshold_default` | float | no | Default `0.7`. |

## Pose-estimation fields (`pose_estimator` role)

Fields that apply when `role` includes `pose_estimator`.

### `outputs` (pose)

Additional fields on pose output tensors:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `keypoint_axis` | int | conditional | Axis holding the keypoints in a pose output. Default `1` for `(N, K, …)` tensors (`simcc_x`/`simcc_y`/`simcc_z`/`heatmap`/`keypoints`); required (`2`) for `poses` `(1, N, K, …)`. |
| `keypoint_count` | int | no | Number of keypoints in a pose output. When authored, must equal `len(tracked_points)`; when omitted, it is derived as `len(tracked_points)`. |

For `semantic: poses` (bottom-up), the tensor is `(1, N, K, 3)` — batch, instances (`N`), keypoints (`K`), and `(x, y, score)` per keypoint — with `keypoint_axis: 2`. When `pose.decode.is_3d: true`, it is `(1, N, K, 4)` — `x, y, z, score` per keypoint, with `z` root-relative depth in `pose.decode.depth_unit`.

### `pose` (pose estimator)

The pose output contract declares the keypoints a model produces, the skeleton(s) joining them, how they map onto the canonical (VRM 1.0) skeleton, and how raw outputs are decoded into keypoints.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pose.estimator_type` | enum | yes | `top_down_single_person`, `top_down_multi_person`, or `bottom_up`. Top-down models see one person per crop (single-person outputs); bottom-up models (e.g. RTMO) emit all persons in one pass. |
| `pose.requires_detector` | bool | yes | `true` when the model consumes a person crop produced by an upstream detector (top-down models — the host warps each detection box via `affine_person_crop`); `false` when the model consumes the full image (bottom-up models such as RTMO). Must be consistent with `estimator_type`. |
| `pose.landmark_schema` | string | no | Keypoint convention the model uses (e.g. `COCO_WholeBody`, `COCO17`, `MediaPipe_Hand`, `MediaPipe_Pose`, `LaPa_106`). |
| `pose.tracked_points` | array[string] or `$ref` | yes | Ordered list of keypoint names the model outputs. The i-th name is the i-th model output index. |
| `pose.connections` | array[Skeleton] or `$ref` | yes | Named skeletons; each joins `tracked_points` into a different skeleton. |
| `pose.derived_points` | object | no | Optional map of derived point name → derivation. May be referenced by `connections` edges. |
| `pose.canonical_mapping` | object or `$ref` | no | Maps model keypoint names onto the canonical (VRM 1.0) skeleton (see [Canonical skeleton](#canonical-skeleton)). |
| `pose.decode` | object | yes | How raw model outputs are decoded into keypoints (see below). Required for all `pose_estimator` models. |

```yaml
pose:
  estimator_type: top_down_single_person
  requires_detector: true
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
  [prefix]wrist: "root"
  [prefix]ring_finger_tip: "ring_finger4"
```

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
| `pose.decode.coordinate_scale` | enum | no | `coordinate` only: scale of the `(N, K, D)` output — `pixel` (raw pixels in the input image; default), `normalized_01` ([0, 1] fraction of the input), or `normalized_11` ([-1, 1] centered on the input). |
| `pose.decode.is_3d` | bool | no | Default `false`. When `true`, the model emits a third coordinate: a `simcc_z` output (simcc), `(N, K, 3)` coordinates (`coordinate`), or a `poses` tensor `(1, N, K, 4)` (bottom-up `coordinate`). 3D `heatmap` is **not supported**. |
| `pose.decode.depth_unit` | enum | required when `is_3d: true` | Unit of the root-relative depth coordinate: `m` (RTMW3D), `mm`, or `pixel` (MediaPipe, image-width-scaled). |
| `pose.decode.depth_range` | array[2] | required when `is_3d: true` and `method: simcc` | `[min, max]` root-relative depth (in `depth_unit`) that `Z_bins` span; the decoded bin position maps linearly to depth. |

Decode methods:

- `simcc` — one `simcc_x` `(N, K, X_bins)` and one `simcc_y` `(N, K, Y_bins)` output (plus `simcc_z` `(N, K, Z_bins)` when `is_3d: true`). The bin counts are read from the output tensors; the bin→pixel scale is `input_width / X_bins` and `input_height / Y_bins` (see Shape key). Decoded keypoints are in cropped image space; the host unprojects them to the source image. The `z` axis maps `Z_bins` linearly onto `depth_range` (in `depth_unit`) when `is_3d: true`.
- `heatmap` — one output with `semantic: heatmap`, shape `(N, K, H, W)` (2D only); per-keypoint peaks are decoded to locations in cropped image space, then unprojected to the source image.
- `coordinate` — coordinates are already decoded (no argmax) and live in the scale of the model's input image, per `coordinate_scale`; the host unprojects them to the source image. Two forms:
  - single-person — one output with `semantic: keypoints`, shape `(N, K, 2)` (`x, y`) or `(N, K, 3)` (`x, y, z`) when `is_3d: true`; `z` is root-relative depth (in `depth_unit`).
  - bottom-up — a `detections` output plus a `poses` output (see the bottom-up note below); `z` is root-relative depth (in `depth_unit`) when `is_3d: true`.

Bottom-up models (`bottom_up`, e.g. RTMO) work on the full image and decode via `coordinate`: a `detections` tensor `(1, N, 5)` — `x1, y1, x2, y2, score` per instance — and a `poses` tensor `(1, N, K, 3)` — `x, y, score` per keypoint (or `(1, N, K, 4)` — `x, y, z, score` — when `is_3d: true`), both in the scale of the model's input image. `N` is the fixed number of decoded instances. The host runs box NMS over the `detections` tensor (declared via detection `decode.requires_nms`) before emitting the surviving instances' keypoints.

Shape key:

- `N` — batch size (person crops for `top_down_multi_person`; detected instances for `bottom_up`).
- `K` — number of keypoints (`len(tracked_points)`).
- `H`, `W` — spatial height and width of the output tensor.
- `X_bins`, `Y_bins` — SIMCC x/y bin counts, read from the output tensors; the bin→pixel scale is `input_width / X_bins` and `input_height / Y_bins`.
- `Z_bins` — SIMCC depth bin count (`is_3d: true` only); bins span `pose.decode.depth_range` linearly.
- For `is_3d: true`, the third coordinate is root-relative depth (RTMW3D), in `depth_unit`. Note: `(N, K, 3)` is `x, y, z` for 3D single-person keypoints, `x, y, score` for bottom-up `poses` `(1, N, K, 3)`, and `(1, N, K, 4)` is `x, y, z, score` for 3D bottom-up `poses`.

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

A family ships several sizes of the same model — e.g. `yolo26` ships `nano`, `small`, and `medium` — all declared in a single `yolo26.yaml`. Sizes share the **output contract** (`outputs` semantics, `decode`, and `normalization`) and differ only in input resolution and the ONNX artifacts they point to:

```yaml
sizes:
  nano:
    input:
      shape: [-1, 3, 640, 640]
      resize:
        target_size: [640, 640]
    onnx:
      batch_artifacts:
        2:
          precision_artifacts:
            fp32:
              filename: yolo26-nano_b2_fp32.onnx
            fp16:
              filename: yolo26-nano_b2_fp16.onnx
  small:
    input:
      shape: [-1, 3, 1024, 1024]
      resize:
        target_size: [1024, 1024]
    onnx:
      batch_artifacts:
        2:
          precision_artifacts:
            fp32:
              filename: yolo26-small_b2_fp32.onnx
            fp16:
              filename: yolo26-small_b2_fp16.onnx
```

Each size entry deep-merges over the shared top-level contract (see [Sizes](#sizes)).

## Normalization modes

`input.normalization` declares how the host transforms letterboxed pixels before `session.run()`. Use a closed enum string, or a `custom` object.

| Mode | Host behavior | Typical use |
|------|---------------|-------------|
| `none` | Letterbox → pass `uint8` pixels unchanged | YOLOX — ONNX graph normalizes internally |
| `unit_float` | `pixel.astype(float) / 255.0` per channel → `[0, 1]` | YOLO26 fp32/fp16 — Ultralytics-style float input |
| `imagenet_bgr` | `(pixel - mean) / std` on 0–255 BGR pixels; mean `(123.675, 116.28, 103.53)`, std `(58.395, 57.12, 57.375)` | RTMPose pose sidecars (BGR input) |
| `imagenet_rgb` | `(pixel - mean) / std` on 0–255 RGB pixels; mean `(123.675, 116.28, 103.53)`, std `(58.395, 57.12, 57.375)` | Pose exports that normalize RGB input |
| `custom` | `(pixel * scale - mean) / std` on 0–255 channel values | Rare exports that match no named mode |

Normalization is applied after letterbox (after the affine warp for `affine_person_crop`; directly on the source image for `method: none`). The `imagenet_*` modes apply ImageNet mean/std in the channel order named by the mode — `imagenet_bgr` on BGR pixels, `imagenet_rgb` on RGB pixels. `none` and `unit_float` are channel-agnostic. `custom` declares its own channel order via `color_format` (default `rgb`).

**Named mode (preferred):**

```yaml
input:
  normalization: unit_float
```

**Custom mode:**

```yaml
input:
  normalization:
    mode: custom
    color_format: rgb            # optional; rgb (default) or bgr — order of mean/std
    scale: 0.00392156862745098   # optional; default 1.0
    mean: [0.0, 0.0, 0.0]
    std: [1.0, 1.0, 1.0]
```

**Per-precision overrides:**

```yaml
input:
  normalization: unit_float
  normalization_by_precision:
    int8: none
```

Resolution order for a selected precision: `normalization_by_precision[precision]` → top-level `normalization` → default `imagenet_bgr`. `normalization_by_precision` values are named mode strings only; a `custom` normalization cannot be a per-precision override — declare it at the top-level `normalization` instead.

> ⚠️ The `imagenet_bgr` default is the RTMPose convention. Detector models must declare `unit_float` or `none` explicitly — relying on the default yields incorrect normalization for most detectors.

**Validation rules:**

- `normalization` defaults to `imagenet_bgr` when omitted.
- String values must be one of `none`, `unit_float`, `imagenet_bgr`, `imagenet_rgb`.
- Object form must have `mode: custom`; `mean`/`std` are length-3 sequences when present; `scale` is a positive number when present; `color_format` (when present) is `rgb` or `bgr`.
- `normalization_by_precision` keys must be a subset of `fp32`, `fp16`, `int8`; values must be valid mode strings.

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

## Validation rules

A sidecar is valid when all of the following hold:

- [ ] `schema_version` is a string matching the skellytracker version pattern (no pre-release tag).
- [ ] `schema_version` is supported by the installed skellytracker release (`installed >= schema_version`).
- [ ] `model_id` matches the sidecar basename (`{model_id}.yaml`).
- [ ] `role` is a non-empty subset of `detector`/`pose_estimator`.
- [ ] `display_name`, `input` (with `name`), and a non-empty `outputs` array (each with `name`) are present.
- [ ] A sidecar whose `role` includes `detector` declares `decode`.
- [ ] A sidecar whose `role` includes `pose_estimator` declares `pose` (with `estimator_type` and `requires_detector`), `pose.decode`, `pose.tracked_points`, and `pose.connections`.
- [ ] `pose.requires_detector` is a boolean and consistent with `pose.estimator_type`: `true` for `top_down_single_person`/`top_down_multi_person`, `false` for `bottom_up`.
- [ ] `pose.estimator_type` is one of `top_down_single_person`, `top_down_multi_person`, `bottom_up`; `pose.decode.method` is one of `simcc`, `heatmap`, `coordinate`.
- [ ] A sidecar with `pose.requires_detector: true` declares `input.resize.method: affine_person_crop`.
- [ ] `pose.tracked_points` names are unique.
- [ ] Every `pose.connections[].edges` endpoint appears in `pose.tracked_points` or `pose.derived_points`.
- [ ] `pose.connections[].name` values are unique.
- [ ] `pose.derived_points` names are unique; each derivation is a string (passthrough), a list of names (mean), or a name→number map (weighted), and every referenced source name appears in `pose.tracked_points`.
- [ ] `pose.canonical_mapping` (when present) has unique keys; each value is a string (passthrough), a list of names (mean), or a name→number map (weighted), and every referenced source name appears in `pose.tracked_points`.
- [ ] `sizes` is present with at least one size.
- [ ] Each size declares `input.shape` with `-1` as its batch axis (first dimension).
- [ ] Each size's `input.shape` spatial dims equal its `resize.target_size` when `resize` is present, `method` is `letterbox` or `affine_person_crop`, and `supports_dynamic_size` is not set. `target_size` is omitted when `method: none` or `supports_dynamic_size: true`; spatial dims are `-1` when `supports_dynamic_size: true`.
- [ ] `input.resize.target_size` (when present) is a length-2 `[height, width]` array.
- [ ] `$ref` directives resolve to existing files, contain only the `$ref` key, and terminate (no cycles).
- [ ] `base` directives resolve and merge (with `null`-delete), and the derived file overrides `model_id` and `schema_version`.
- [ ] `batching.batch_axis` (when present) is `0` (non-zero batch axes are out of scope).
- [ ] `input.dtype` is present; its keys are a subset of `fp32`/`fp16`/`int8` covering the precisions declared in `batch_artifacts`, and its values are ONNX element types (`float32`, `float16`, `uint8`, `int8`). `input.layout` (when present) is `NCHW` or `NHWC`.
- [ ] Each output's `dtype` is present; its keys are a subset of `fp32`/`fp16`/`int8` covering the precisions declared in `batch_artifacts`, and its values are ONNX element types (`float32`, `float16`, `uint8`, `int8`).
- [ ] On each output, `rank` (when present) equals `len(shape)`; `rank` is omitted when `shape` is omitted.
- [ ] `input.normalization` (when present) is a valid mode (`none`, `unit_float`, `imagenet_bgr`, `imagenet_rgb`, or `custom` object); when omitted, it defaults to `imagenet_bgr`. A `custom` object has `mode: custom`, length-3 `mean`/`std` when present, a positive `scale` when present, and `color_format` `rgb`/`bgr` when present.
- [ ] `normalization_by_precision` keys are a subset of `fp32`/`fp16`/`int8`, and its values are valid mode strings (`none`, `unit_float`, `imagenet_bgr`, `imagenet_rgb`) — never a `custom` object.
- [ ] `input.resize.method` is `letterbox`, `affine_person_crop`, or `none`, and `input.resize.interpolation` is `linear`, `area`, `cubic`, or `nearest`, when `input.resize` is present.
- [ ] `input.resize.crop_policy` is present only when `method: affine_person_crop`; `pad_value` and `preserve_aspect_ratio` are present only when `method: letterbox`.
- [ ] `input.resize.crop_policy.expand_ratio` (when present) is a positive float, and `crop_policy.maintain_aspect_ratio` (when present) is a boolean.
- [ ] Each size's `onnx.batch_artifacts` has positive native batch keys or a single `dynamic` key (never mixed), and non-empty per-group `precision_artifacts`.
- [ ] Precision keys are the closed enum `fp32`/`fp16`/`int8`.
- [ ] Each precision artifact declares `filename`.
- [ ] A precision artifact's `input_dtype` (when authored) agrees with `input.dtype[precision]`.
- [ ] When a precision artifact declares `url`, it also declares `url_sha256`.
- [ ] When a size has `len(batch_artifacts) > 1`, each group declares `output_shapes`, whose length equals `len(outputs)` (parallel to `outputs` order) and whose entries are arrays.
- [ ] All sizes share the same output semantics and keypoint count (spatial-dependent dims may scale with input size).
- [ ] `keypoint_count` (when authored on a pose output) equals `len(pose.tracked_points)`.
- [ ] Each output's `semantic` is one of `detections`, `simcc_x`, `simcc_y`, `simcc_z`, `heatmap`, `keypoints`, `poses`. `detections` requires `role` to include `detector`; every pose semantic requires `role` to include `pose_estimator`.
- [ ] A sidecar whose `role` includes `detector` declares exactly one output with `semantic: detections`, and that output declares `fields`.
- [ ] A `detector`'s `decode.score_field`/`decode.class_field` (when authored) appear in the `detections` output's `fields`; `decode.box_format` (when present) is `xyxy`, `xywh`, or `cxcywh`.
- [ ] A `simcc` decoder declares exactly one `simcc_x` and one `simcc_y` output, plus one `simcc_z` output iff `is_3d: true`; `heatmap` declares exactly one output with `semantic: heatmap`; `coordinate` declares exactly one `keypoints` output (single-person), or one `detections` + one `poses` output (`bottom_up`).
- [ ] A `poses` output declares `keypoint_axis: 2`.
- [ ] `pose.decode.is_3d` (when present) is a boolean; when `true`, `pose.decode.depth_unit` is present and one of `m`, `mm`, `pixel`; `depth_unit` and `depth_range` are present only when `is_3d: true`.
- [ ] When `pose.decode.is_3d: true` and `pose.decode.method: simcc`, `pose.decode.depth_range` is present and is a length-2 array `[min, max]` in `depth_unit`.
- [ ] `pose.decode.method: heatmap` with `pose.decode.is_3d: true` is invalid (3D heatmap is not supported).
- [ ] `pose.decode.coordinate_scale` is present only when `pose.decode.method: coordinate`, and (when present) is `pixel`, `normalized_01`, or `normalized_11`.
- [ ] `overlay.skeleton` (when a string) names an entry in `pose.connections`; when an inline edge list, every endpoint appears in `pose.tracked_points` or `pose.derived_points`.
- [ ] Each `overlay.groups.<name>` declares `connection_color` and uses `connections` or `prefix` (not both); at most one group has neither (the default group).
- [ ] `overlay.groups.<name>.connection_thickness` (when present) is a positive integer.
- [ ] Every edge in a group's `connections` is a member of the selected skeleton's edge set (the named `pose.connections[]` entry, or the inline `overlay.skeleton` edge list).
- [ ] `overlay.keypoint_color`, `overlay.groups.<name>.connection_color`, and `overlay.groups.<name>.keypoint_color` (when present) are length-3 arrays.
- [ ] A sidecar whose `role` includes `detector`: `decode.person_class_id` >= `decode.class_id_base`; `decode.confidence_threshold_default` (when present) is in `[0, 1]`; `decode.max_detections` (when present) is a positive integer; `decode.class_id_base` and `decode.person_class_id` are non-negative integers.

Every future contract change must bump `schema_version` to the skellytracker release that ships it and add a changelog row below.

## Changelog

| `schema_version` | Changes |
|------------------|---------|
| `v2024.09.1019` | Initial YAML sidecar spec: one `{model_id}.yaml` per family with a `sizes` map (size × batch × precision via `sizes.<size>.onnx.batch_artifacts`, incl. `dynamic` batch); `role` arrays (`detector` / `pose_estimator`, both for one-stage models); `input.normalization` modes (`none`, `unit_float`, `imagenet_bgr`, `imagenet_rgb`, `custom`) with per-precision overrides; `input.resize` (`letterbox` + `affine_person_crop` with `crop_policy`, `none`, interpolation); file composition (`$ref` includes + `base` inheritance with `null`-delete); `schema_version` as skellytracker version string; pose output contract (`tracked_points`, named `connections`, `derived_points`, `canonical_mapping` with `prefixes`, `overlay.groups`, `requires_detector`, and `decode` incl. `simcc`/`heatmap`/`coordinate` with 3D via `is_3d`/`depth_unit`/`depth_range`); VRM 1.0 canonical skeleton with current→VRM name-migration note; `detectors/` storage layout; per-precision `url` with mandatory `url_sha256`. |

## Reference examples

### Complete YOLO26 detector sidecar (single size)

A family sidecar with one `nano` size, three precisions, and per-precision checksums:

```yaml
---
schema_version: "v2024.09.1019"
model_id: yolo26
display_name: YOLO26
role: [detector]

input:
  name: images
  dtype:
    fp32: float32
    fp16: float16
    int8: uint8
  layout: NCHW
  normalization: unit_float
  normalization_by_precision:
    int8: none
  resize:
    method: letterbox
    preserve_aspect_ratio: true
    pad_value: 114
    interpolation: linear

batching:
  batch_axis: 0

outputs:
  - name: output0
    dtype:
      fp32: float32
      fp16: float32      # output stays float32 even for fp16 input (Ultralytics export)
      int8: float32      # dequantized output
    shape: [2, 300, 6]
    rank: 3
    semantic: detections
    fields: [x1, y1, x2, y2, score, class_id]

decode:
  box_format: xyxy
  class_id_base: 0
  person_class_id: 0
  score_field: score
  class_field: class_id
  max_detections: 300
  may_include_non_person_classes: true
  requires_nms: false
  confidence_threshold_default: 0.7

sizes:
  nano:
    input:
      shape: [-1, 3, 640, 640]
      resize:
        target_size: [640, 640]
    onnx:
      batch_artifacts:
        2:
          precision_artifacts:
            fp32:
              filename: yolo26-nano_b2_fp32.onnx
            fp16:
              filename: yolo26-nano_b2_fp16.onnx
            int8:
              filename: yolo26-nano_b2_int8.onnx
```

### Multi-batch artifact fragment (one size, two native batches)

```yaml
sizes:
  nano:
    input:
      shape: [-1, 3, 640, 640]
      resize:
        target_size: [640, 640]
    onnx:
      batch_artifacts:
        2:
          precision_artifacts:
            fp32:
              filename: yolo26-nano_b2_fp32.onnx
            fp16:
              filename: yolo26-nano_b2_fp16.onnx
          output_shapes:
            - [2, 300, 6]
        4:
          precision_artifacts:
            fp32:
              filename: yolo26-nano_b4_fp32.onnx
            fp16:
              filename: yolo26-nano_b4_fp16.onnx
          output_shapes:
            - [4, 300, 6]
```

The input shape for a group is derived: `[N] + input.shape[1:]` (batch key `N` + the size's image shape). When a size lists only one batch key, the top-level `outputs[].shape` may be used instead of per-group `output_shapes`.

### RTMW WholeBody pose estimator sidecar (multi-size, dynamic batch, with download URLs)

```yaml
---
schema_version: "v2024.09.1019"
model_id: rtmw-wholebody
display_name: RTMW WholeBody
role: [pose_estimator]

input:
  name: input
  dtype:
    fp32: float32
  layout: NCHW
  normalization: imagenet_bgr
  resize:
    method: affine_person_crop
    interpolation: linear
    crop_policy:
      expand_ratio: 1.25
      maintain_aspect_ratio: true

batching:
  batch_axis: 0

outputs:
  - name: simcc_x
    dtype:
      fp32: float32
    semantic: simcc_x
    keypoint_axis: 1
  - name: simcc_y
    dtype:
      fp32: float32
    semantic: simcc_y
    keypoint_axis: 1

pose:
  estimator_type: top_down_single_person
  requires_detector: true
  landmark_schema: COCO_WholeBody
  tracked_points: {$ref: ../../../shared/skeletons/coco133_tracked_points.yaml}
  connections: {$ref: ../../../shared/skeletons/coco133_skeletons.yaml}
  decode:
    method: simcc

overlay:
  skeleton: rtmpose_skeleton
  keypoint_color: [0, 255, 128]
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
  l-m:
    input:
      shape: [-1, 3, 256, 192]
      resize:
        target_size: [256, 192]
    onnx:
      batch_artifacts:
        dynamic:
          precision_artifacts:
            fp32:
              filename: rtmw-wholebody-l-m_fp32.onnx
              url: https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip
              url_sha256: "<lowercase-hex-sha256>"
  x-l:
    input:
      shape: [-1, 3, 256, 192]
      resize:
        target_size: [256, 192]
    onnx:
      batch_artifacts:
        dynamic:
          precision_artifacts:
            fp32:
              filename: rtmw-wholebody-x-l_fp32.onnx
              url: https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip
              url_sha256: "<lowercase-hex-sha256>"
```
