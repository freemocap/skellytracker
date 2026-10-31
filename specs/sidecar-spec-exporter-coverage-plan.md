# Plan: extend the sidecar spec to cover every exporter model

> **Status:** plan, revised 2026-10-20. Target: [`sidecar-spec.md`](sidecar-spec.md) on `philip/sidecar_implementation`.
> **Base document:** `../Roboflow-Exporter/plans/sidecar-spec-stretch-multi-output.md` (the "draft",
> 2026-10-19). The draft already has replacement text for most of the RF-DETR and RTMDet changes. This
> plan adopts the draft, corrects it where the repos have moved on, trims fields that no exported model
> uses, and adds the gaps it misses.
> **Scope:** `Roboflow-Exporter`, `OpenMMLab-Exporter` and `Ultralytics-Exporter`. `Qualcomm-Exporter` is
> excluded until that library is complete.

## Context

| Repo | Families | Fits current spec? |
|------|----------|--------------------|
| `Roboflow-Exporter` | `rfdetr`: nano 384, small 512, medium 576, large 704; b1–3; fp32/fp16 | **No.** Needs stretch resize, a box tensor separate from a class-logit tensor, and a query×class top-k. The library will write sidecars once the spec supports these. |
| `OpenMMLab-Exporter` | `rtmdet-person` (nano 320, medium 640); `rtmdet` (CSPNeXt: tiny, small, medium, large, xlarge 640, x-p6 1280); **new family** for the RGB-backbone sizes (l-convnext-b, l-swin-b 640, l-swin-b-p6 1280); `rtmw-wholebody` (5 sizes, not exported yet) | **No.** Needs a separate class-id tensor, letterbox placement and rounding, and explicit NMS parameters. |
| `Ultralytics-Exporter` | `yolo26` detector, `yolo26-body` pose (nano…xlarge 640, b1–3, fp32/fp16/int8) | Detector yes, apart from channel order and letterbox placement. **Pose no:** a packed `[N,300,57]` tensor is declared twice, as `detections` and as `poses`. |

## Per-repo findings

### Roboflow-Exporter (RF-DETR) — verified against code and the 24 exported graphs

Each claim was checked against the `.onnx` files under `exports/.../rfdetr/` and the decode and
preprocessing code:

| Claim | Evidence | Result |
|-------|----------|--------|
| Input `input` `[B,3,S,S]` float32, S = 384/512/576/704, static B = 1/2/3 | ONNX graph inputs, all 24 artifacts | ✅ |
| Outputs `dets` `[B,300,4]` float32, `labels` `[B,300,91]` float32 | ONNX graph outputs | ✅ |
| fp16 artifacts keep float32 I/O | fp16 graphs' I/O types | ✅ |
| **No in-graph normalization**; the host feeds ImageNet-normalized RGB | `input` → `Cast` → patch-embed `Conv` (no Sub/Div); `onnx_runtime/preprocess.py` | ✅ |
| No decode in the graph | The only `TopK` is `/transformer/TopK` (K=300), the two-stage encoder's query selection. `dets`/`labels` come straight from the heads. No `NonMaxSuppression`. | ✅ |
| Stretch resize, bilinear, half-pixel centres, **no antialias** | `_bilinear_resize_half_pixel` (matches torchvision `antialias=False` and cv2 `INTER_LINEAR`) | ✅ |
| `dets` are normalized `cxcywh` | Decoder scales by the source `(W,H)` and converts to `xyxy` | ✅ |
| Decode = sigmoid → stable top-300 over Q×C (ties broken by ascending flat index) → `query = i // C`, `class = i % C` → clamp to the image | `convert_rf_detr_coco.py::_select_topk_multiclass`, which uses `lexsort` | ✅ |
| Threshold applied after top-k | `scores > threshold`, **strict**. Evaluation uses 0.0; `confidence_threshold_default = 0.5` comes from `identity.py`. | ✅ (strict `>` noted) |
| 91 slots; slot index = COCO category id; slot 1 = person | `category_id = class_ids[label]` with `class_ids = arange(91)` | ✅ |
| `background_class_id=None` (no slot excluded before top-k) | `_exclude_background_class` | ✅ |

Notes for the spec text:

- **R1.** The reference decoder does **not drop** slot 0 or the ten COCO id gaps. It emits them as unknown COCO category ids, which pycocotools never matches. A `class_id_map` with `-1` for those slots, applied *after* top-k, therefore gives identical COCO scores while hiding meaningless classes from the host. This is the draft's `class_id_map_before_selection: false` behaviour.
- **R2.** The exporter matches upstream `RFDETR.predict()` (`F.resize(..., antialias=False)`). Upstream's own COCO val dataset resizes through PIL, which always antialiases. Published RF-DETR AP numbers therefore use a slightly different resize than the deployable contract. That's fine, but worth a note in reports.
- **R3.** No sidecar writer exists yet. `exports/.../rfdetr.yaml` is a gitignored leftover from the removed pipeline. It is non-spec and wrongly claims `normalization.location: in_graph`, so don't use it as a reference.
- **R4.** The draft's §5.1 example is correct except that its batch keys aren't quoted (`1:`), which the current spec rejects.

### OpenMMLab-Exporter (RTMDet, RTMW)

- **O1.** Outputs are `dets` `[N,Q,5]` (`x1,y1,x2,y2,score`) and `labels` `[N,Q]` int64. The sidecar writer drops `labels` (`_declared_output_names`) and writes the `dets` shape as `[-1,-1,-1]`; the last dimension should be `5`.
- **O2.** The CSPNeXt sizes normalize **BGR** (`to_rgb=False`), while `l-convnext-b`, `l-swin-b` and `l-swin-b-p6` set `bgr_to_rgb=True` (**RGB**). **Decision: the RGB-backbone sizes become their own family**, so every family has one channel order. The name is still open; `rtmdet-rgb` is a placeholder. All three COCO families inherit the same `test_cfg` (score 0.001, IoU 0.65, `max_per_img` 300, `nms_pre` 30000).
- **O3.** Preprocessing and benchmark inconsistencies. See [OpenMMLab export and benchmark flow](#openmmlab-export-and-benchmark-flow) for the investigation and the fix.
- **O4.** **The in-graph candidate caps come from a misplaced config key.** The deploy configs set a top-level `post_processing=dict(pre_top_k=-1, max_output_boxes_per_class=100000)`. mmdeploy reads only `codebase_config.post_processing` (`get_post_processing_params`), so the `base_static.py` defaults (`pre_top_k=5000`, `max_output_boxes_per_class=200`) are baked in instead. That explains the draft's measured TopK 5000 and 200-per-class cap. The rest of what the draft measured (IoU 1.0, score 0.0, `keep_top_k` 100000) comes from the `_no-nms` model config through `rtmdet_head.py`.
- **O5.** RTMW height and width are transposed in the registry; see [RTMW height/width swap](#rtmw-heightwidth-swap).
- **O6.** RTMW otherwise fits the current spec: `affine_person_crop` (bbox ×1.25, aspect fixed to w/h, border 0), `imagenet_rgb`, SIMCC split ratio 2.0 (the spec's `input_width / X_bins` rule is equivalent), plain argmax.
- **O7.** INT8 export was retired (`443a007`), but `*_int8.onnx` files are still on disk and still listed in the sidecars.

### Ultralytics-Exporter (YOLO26)

- **U1.** The detector (`[N,300,6]`, NMS-free one-to-one head, graph top-300) fits the draft as `selection: none`, `max_detections: 300`. No new output semantics are needed.
- **U2.** **Packed pose is not representable.** `yolo26-body` emits a single `output0` of `[N,300,57]` = 6 detection columns + 17×(x, y, score). The sidecar lists `output0` twice: as `detections` with 6 `fields` (but the last dimension is 57), and as `poses` with rank 3 (the spec requires `(1,N,K,3)` rank 4). Duplicate output names aren't rejected by any current rule. **A new spec construct is needed (S6).**
- **U3.** Letterbox: **centre** placement; new size = Python `round` (**half-even**); top/left = `round(d/2 − 0.1)`, which is `floor(d/2)`; cv2 `INTER_LINEAR`; pad 114; BGR→RGB, then `/255`. The sidecar can't say centre, rounding or RGB today.
- **U4.** int8 artifacts take **float32** input. The spec's YOLO26 reference example (`int8: uint8` + `normalization_by_precision: {int8: none}`) doesn't match a real export.

**Rounding rules in use:** two letterbox implementations use different rules: half-even (Ultralytics) and half-up (mmdet `_scale_size`, `int(x·s + 0.5)`). A default would be wrong for one of them, so `rounding` must be **required** for letterbox.

## RTMW height/width swap

mmpose `input_size` and mmdeploy `onnx_config.input_shape` are both **(width, height)**. For
`rtmw-*-256x192` that means W=192, H=256, so the ONNX input is `[N,3,256,192]`. The 384×288 sizes are `[N,3,384,288]`.

The mistake, and everywhere it propagates:

| Location | What it does | Effect |
|----------|--------------|--------|
| `exporter/models.py` `RTMW_MODELS` (lines 290–330) | `input_height=192, input_width=256` (and `288, 384`): mmpose's `(w, h)` read as `(h, w)` | **Root error** |
| `exporter/worker/export_fp32.py:66` | `cfg.onnx_config.input_shape = (img_h, img_w)` even though mmdeploy expects `(w, h)` (see `process_model_config`'s docstring) | **Second error that cancels the first**: it passes `(192, 256)`, which mmdeploy reads as W=192, H=256, so the exported graph should come out correctly by accident. Fixing only `models.py` would break the export, so fix both together. For square RTMDet inputs neither error shows. |
| `RtmModelIdentity.input_shape` → `exporter/sidecar.py::_sizes_block` | Writes `shape: [-1,3,192,256]` and `target_size: [192,256]` | Wrong sidecar. Latent, because no RTMW sidecar has been written yet (no RTMW exports). |
| `exporter/preprocess.py` (`fallback_preprocessing`, `_from_pipeline_json`) → `benchmark.py` | Builds `192×256` inputs | The FPS benchmark would feed the wrong spatial shape to a static `256×192` graph and fail in onnxruntime. |
| Draft §5.4 (`Roboflow-Exporter/plans/sidecar-spec-stretch-multi-output.md`) | Copies the registry: `shape: [-1,3,192,256]`, `target_size: [192,256]`, `[-1,3,288,384]` | Wrong reference example. The spec should use `[-1,3,256,192]` / `[256,192]` and `[-1,3,384,288]` / `[384,288]`. |

Fix: store `(input_width, input_height)` correctly in the registry (`input_height=256, input_width=192`),
pass `(img_w, img_h)` to `onnx_config.input_shape`, and add a test that loads an exported graph and asserts
`input.shape[2:] == (identity.input_height, identity.input_width)`. This is not verified on a real RTMW
graph yet, because none has been exported; the analysis comes from the mmpose/mmdeploy conventions.

## OpenMMLab export and benchmark flow

### What actually happens today

| Step | Preprocessing | Post-processing | Category ids |
|------|---------------|-----------------|--------------|
| PyTorch baseline (`eval_pth.py` → mmdet `tools/test.py`, baseline config) | Model config `test_pipeline`: `Resize keep_ratio=True` (mmcv half-up rounding, cv2 bilinear) + `Pad` to `S×S` bottom/right with 114, i.e. a **top-left letterbox**; BGR or RGB per `data_preprocessor` | mmdet `test_cfg`: per-level `nms_pre`, `score_thr`, class-wise NMS, `max_per_img` | `evaluate_pkl`: `label + 1` |
| ONNX eval (`eval_onnx.py` → mmdeploy task processor) | `build_dataset` uses the **same `test_pipeline`, unchanged** (mmdeploy `task.py:138`), so this is also a **letterbox** | Graph: TopK 5000, NMS IoU 1.0 / 200 per class (O4). Host: `_consumer_nms` **hardcoded to 0.05 / 0.6 / 100 for every family** | `label + 1` |
| SDK `pipeline.json` (mmdeploy `export2SDK` → `process_model_config(input_shape=…)`) | Rewritten to `Resize keep_ratio=False`, i.e. **stretch** | — | — |
| Sidecar (`sidecar.py` reads `pipeline.json`) | Declares **stretch** (`letterbox` + `preserve_aspect_ratio: false`) | `requires_nms: true`, no thresholds | — |

So the two benchmarks agree with each other: both letterbox. The inconsistency is that **the sidecar
describes a different preprocessing (stretch) from the one that was benchmarked**, because it reads
mmdeploy's SDK pipeline, which the repo doesn't otherwise use.

The investigation turned up two more defects:

1. **COCO category mapping bug.** `evaluate_pkl` writes `category_id = label + 1`. That's correct only for person-only models. For COCO-80, mmdet labels 0–79 map to the sparse ids 1–90, so every label ≥ 11 is scored as the wrong class. Every published COCO-80 RTMDet AP is invalid: Tiny is reported at 0.066 (PyTorch) and 0.005 (ONNX) against a model-zoo AP of about 0.41. Person-only results are unaffected (`rtmdet-person-nano` 0.349 / 0.348).
2. **The reported ONNX AP has a further, unexplained loss.** The 10-17 reports show ONNX 0.005 against PyTorch 0.066 for Tiny. Reproducing the `label + 1` bug directly on the same artifact gives 0.067 (table below). The extra drop is in the mmdeploy ONNX eval path, not the graph. Reading `End2EndModel`, the data-preprocessor updater and the configs found nothing that explains it. The eval worker has changed since those reports (`b6d7b1a`), so re-run before chasing it, or retire that path as proposed below.

**Experiment** (`rtmdet-tiny_b1_fp32.onnx`, first 500 COCO val2017 images, onnxruntime CPU, cv2,
faster-coco-eval; scripts in the session scratchpad):

| Preprocessing | NMS | Category ids | AP | AP50 |
|---------------|-----|--------------|----|------|
| letterbox (mmdet rules) | baseline 0.001 / 0.65 / 300 | correct (`cat_ids[label]`) | **0.449** | 0.615 |
| letterbox | baseline | `label + 1` (current code) | 0.067 | 0.096 |
| letterbox | consumer 0.05 / 0.6 / 100 | correct | 0.448 | 0.616 |
| stretch (current sidecar) | baseline | correct | 0.437 | 0.603 |

Takeaways:

- The exported graph is healthy.
- A host that follows today's sidecar (stretch) loses about 1.1 AP against the letterbox the model was evaluated with.
- The hardcoded NMS settings cost about 0.1 AP here, but they are still wrong in principle.
- The effect of the graph caps (O4) is not measured yet; that needs a re-export.

### Recommended changes

1. **Derive the sidecar from the baseline model config, not `pipeline.json`.** The PyTorch reference is mmdet's `test_pipeline` + `data_preprocessor` + `test_cfg`, so emit:
   - `resize: {method: letterbox, pad_position: top_left, rounding: half_up, pad_value: 114, interpolation: linear}`;
   - `color_format` from `bgr_to_rgb`;
   - `decode.nms` from `test_cfg`: `score_thr` → `score_threshold`, `nms.iou_threshold`, `max_per_img` → `post_nms_top_k`, `min_bbox_size`.

   Fail the export if `test_pipeline` contains a transform the sidecar can't express. Keep `pipeline.json` for provenance only; mmdeploy's SDK stretch rewrite is SDK behaviour the repo doesn't ship.
2. **Fix the export config so the graph emits a complete candidate set:**
   - Move `post_processing` under `codebase_config` in every `deploy_cfgs/detection_*.py`. Use `pre_top_k=-1`, and set `max_output_boxes_per_class` to at least the baseline `nms_pre`.
   - In each `_no-nms` model config, set `score_thr` to the **baseline** `score_thr` (0.001 COCO, 0.05 person) rather than 0.0. With IoU 1.0 the graph still suppresses nothing, but the output shrinks from every anchor × class to the baseline's candidate set.
   - After re-export, assert by graph inspection that no `TopK` with K=5000 and no `max_output_boxes_per_class=200` constant remains.

   The sidecar is then `selection: nms` with no `graph_candidate_limit`.

   The other option is to bake the baseline NMS into the graph (`selection: none`). That's simpler for the host, but it fixes the thresholds in the artifact. I don't recommend it, because skellytracker owns NMS.
3. **Fix the scorer:** map `label → coco_gt.getCatIds()[label]` (sorted) in `evaluate_pkl`, or take `bbox_mAP` straight from mmdet's `CocoMetric`, which `tools/test.py` already runs. Add a guard that fails the run when the PyTorch baseline AP is more than about 0.01 from the model-zoo AP stored in `RtmModelIdentity`; that guard would have caught the 0.066.
4. **Replace the mmdeploy ONNX eval with a sidecar-driven evaluator.** Use plain onnxruntime + cv2 + numpy that reads the generated sidecar and implements exactly its preprocessing, unprojection and `decode.nms` (like the scratch `rtmdet_infer.py`). The ONNX number then measures what a skellytracker host gets, and the evaluator no longer depends on mmdeploy or Docker (it can use the CUDA EP on the host). Keep the PyTorch baseline in Docker. Delete `_consumer_nms`; its constants come from the sidecar instead. Once M6 lands in skellytracker, swap in its generic decoder, or keep a reference copy under a lockstep rule as `Roboflow-Exporter/rf_detr/DECODE_RULES.md` does.
5. **Add a per-image agreement check alongside AP.** Have both passes dump post-NMS detections for the first ~200 images. Report the fraction of PyTorch detections matched by an ONNX detection of the same class at IoU ≥ 0.9, plus the mean score difference. AP can hide offsetting errors; per-image agreement localizes them to preprocessing, the graph or decode.
6. **Acceptance gate per artifact:** fp32 |ΔAP| ≤ 0.005 and fp16 ≤ 0.01 against the PyTorch baseline at the same batch size, plus per-image agreement ≥ 0.98 for fp32. Record both in `report.json`.
7. Make the FPS benchmark (`preprocess.py`) use the same sidecar-driven preprocessing; it currently uses PIL `BILINEAR` (antialiased) with half-even rounding. Fix `report.json`'s `converted.full_image_count`, which currently holds the 200-image FPS subset.

The remaining expected gap after these changes is mmdet's **per-feature-level** `nms_pre` top-k, which a flat
ONNX output can't reproduce. With the score threshold applied first, few levels exceed 30000 candidates,
so the gap should be small; the gate in step 6 will measure it.

## Spec changes, in edit order

Each item says where its text comes from. "Draft §x" means adopt that section's replacement text with the stated amendments.

### S1 — `input.resize` (draft §1.1–1.4, amended)

- Add `method: stretch` (draft §1.2) with the unprojection table.
- Letterbox: add `pad_position` (`top_left` | `center`) and `rounding` (`half_even` | `half_up`). **Both are required at vNEXT.** Drop `floor`/`ceil` (no user).
- **Define the centre offsets:** `left = floor((T_w − n_w)/2)` and `top = floor((T_h − n_h)/2)`, with the remainder on the right/bottom (Ultralytics).
- **Define unprojection as the exact inverse of the resize:** per-axis `n_w/src_w` and `n_h/src_h`, plus the pad offsets. This is what mmdet's `scale_factor` does; Ultralytics' uniform gain differs by under a pixel.
- `pad_value`: allow a scalar or a length-3 array.
- `preserve_aspect_ratio: false` becomes a legacy alias for `stretch` (pre-vNEXT only).
- **Defer `antialias` and `size_divisor`.** Every in-scope reference implementation resizes without antialiasing (cv2 `INTER_*` semantics), and none has dynamic spatial dimensions. State normatively that resizes are not antialiased.

### S2 — `input.color_format` (draft §1.5)

Adopt as written: required when the normalization resolves to `none`/`unit_float`, and it must agree with `imagenet_*`/`custom`. Rewrite `none` as "pixels unchanged in `input.dtype`". This affects the Ultralytics sidecars; RF-DETR and RTMDet use `imagenet_rgb` / `imagenet_bgr`.

### S3 — `affine_person_crop` (draft §1.6, trimmed)

Define the `maintain_aspect_ratio` behaviour (expand the shorter side to `w/h`) and add `crop_policy.pad_value` (default `0`). **Defer `border_mode`**: only `constant` is used. Keep rotation out of scope.

### S4 — Multi-tensor detection outputs (draft §2.1–2.6)

Adopt the semantics `boxes`, `scores`, `class_ids` and `class_logits`, plus `class_axis` and the `decode` bindings (`boxes_output`, `scores_output`, `class_ids_output`, `class_logits_output`), `box_scale` and `clamp_boxes`. Extend the output dtypes with `int32`, `int64` and `bool`. Amendments:

- `score_activation`: `none` | `sigmoid` only. **Defer `softmax`** (no user).
- `class_id_map`: keep it as a length-C array with `-1` meaning ignore, loaded via `$ref` from `detectors/shared/class_ids/coco91.yaml` (RF-DETR). RTMDet doesn't need one: its labels are contiguous (`class_id_base: 0`, `person_class_id: 0`). A `coco80.yaml` map is optional, only for hosts that want to report real COCO category ids.
- **Defer `class_id_map_before_selection`.** Make "map after top-k" the normative behaviour (R1).
- Make every score threshold in the spec a **strict** comparison (`score > threshold`). That covers `confidence_threshold_default` and `nms.score_threshold`. It matches the RF-DETR reference, and mmdet's `filter_scores_and_topk` (from memory; not checked in source here).

### S5 — Selection and NMS (draft §2.3–2.4, §2.7, trimmed)

- Make `decode.selection` (`none` | `top_k` | `nms`) required at vNEXT. `requires_nms` becomes a legacy alias.
- `nms` block: `iou_threshold`, `score_threshold` (required), `class_agnostic`, `post_nms_top_k`. Adopt the stage order from draft §2.4 without the deferred stages.
- **Defer `pre_nms_top_k` and `pre_nms_top_k_scope`.** The only source value, mmdet's `nms_pre`, is a *per-feature-level* top-k. A flat ONNX output can't express that, and putting 1000/30000 into a global `pre_nms_top_k` would describe a different algorithm. Leave it out and let the exporter's parity gate measure the gap.
- **Defer `min_bbox_size`.** Every RTMDet config uses `0`, and mmdet applies it in source-image pixels while the draft defines model-input pixels. Defer it until a model needs a non-zero value.
- **Drop `graph_candidate_limit`.** The RTMDet caps were a config bug (O4) and go away on re-export. Say normatively that the detection tensors are the complete candidate set for the declared `selection`. A graph may pre-filter at `nms.score_threshold` without that being declared, since the result is identical.
- Replace `may_include_non_person_classes` with `is_person_only` (draft).

### S6 — **New:** packed keypoint columns in a `detections` tensor (U2)

Add an optional `packed_keypoints` object to a `semantic: detections` output:

```yaml
outputs:
  - name: output0
    dtype: {fp32: float32, fp16: float32, int8: float32}
    shape: [-1, 300, 57]
    rank: 3
    semantic: detections
    fields: [x1, y1, x2, y2, score, class_id]
    packed_keypoints:
      offset: 6                    # first keypoint column; must equal len(fields)
      components: [x, y, score]    # per-keypoint column order
```

Rules:

- Valid only when `role` includes `pose_estimator`, `estimator_type: bottom_up` and `pose.decode.method: coordinate`.
- `offset == len(fields)` and `offset + K·len(components) == shape[-1]`, where `K = len(tracked_points)`.
- `components` is drawn from {`x`, `y`, `z`, `score`}, must contain `x` and `y`, and contains `z` iff `is_3d`.
- Keypoint coordinates share the box coordinate space (`box_scale`). `score` is already activated.
- The bottom-up `coordinate` decoder accepts **either** `detections` + `poses` (RTMO style) **or** a `detections` output carrying `packed_keypoints`.
- Change the `fields` rule to `len(fields) == shape[-1]` **unless** `packed_keypoints` is present.

### S7 — `max_detections` and bottom-up instance count (draft §3, simplified)

- Adopt the draft's meaning: `max_detections` is only the graph-emitted fixed row count. It is forbidden with `selection: top_k`/`nms`, and there is no default at vNEXT.
- **Don't add `pose.decode.instance_count`.** Instead require that a bottom-up model's `poses` instance axis equals the primary detection row axis, and that pose rows are gathered by the surviving detection row indices. The count is then always derivable.

### S8 — Family invariant (draft §2.8)

Adopt it: a size entry may set only `input.shape`, `input.resize.target_size` and `onnx`. This also removes the current spec's contradiction between "a size is a partial document deep-merged over the top level" and "sizes share `decode`/`normalization`". Add a sentence noting the consequence: sizes that differ in channel order, normalization or NMS contract are separate families. RTMDet now has three: person, CSPNeXt COCO, and RGB-backbone COCO.

### S9 — Validation additions not in the draft

- `outputs[].name` values are unique (catches U2).
- When `shape[-1]` is concrete, a `detections` output's `fields` length matches it, adjusted for `packed_keypoints` (catches O1 once exporters write `5`).
- A letterbox resize declares `pad_position` and `rounding` (vNEXT).

### S10 — Versioning, changelog, examples

- Version-aware legacy rules: adopt draft §4.3, minus the rows for the deferred fields.
- Replace the reference examples with real exports, all with quoted batch keys:
  - YOLO26 detector, fixing `int8: float32` (U4) and adding `color_format: rgb` and `pad_position: center` + `rounding: half_even`;
  - YOLO26 pose with `packed_keypoints` (new);
  - RF-DETR (draft §5.1, quoted keys);
  - RTMDet (draft §5.2/5.3, changed to `letterbox` / `top_left` / `half_up` / `pad_value: 114`, with no `graph_candidate_limit`, `pre_nms_top_k` or `min_bbox_size`);
  - RTMW (draft §5.4 with **`[-1,3,256,192]` / `target_size: [256,192]`** and `[-1,3,384,288]` / `[384,288]`).
- Add one changelog row for vNEXT. Keep the bottom-up separate-tensor example (draft §5.5) as a schema-only example, without `instance_count`.

## Exporter work this unblocks

| Repo | Change |
|------|--------|
| Roboflow | Add a sidecar writer that emits draft §5.1 (`normalization: imagenet_rgb`, `color_format: rgb`, `stretch`, `boxes` + `class_logits`, `selection: top_k`, `top_k: 300`, `coco91` map, `person_class_id: 1`, `confidence_threshold_default: 0.5`). Delete the stale `rfdetr.yaml`. Add a conformance test that loads the YAML with skellytracker's `SidecarModel`. |
| OpenMMLab | Everything in [Recommended changes](#recommended-changes). Emit `labels` as `class_ids` and `dets` with last dimension `5`. Split the RGB-backbone sizes into their own family. Fix the RTMW `(w, h)` handling. Stop listing the retired int8 artifacts. |
| Ultralytics | Detector: add `color_format: rgb`, `pad_position: center`, `rounding: half_even`, `selection: none`. Pose: declare `output0` once, with `packed_keypoints` and `is_person_only: true`. |

## skellytracker implementation impact

All in `skellytracker/core/sidecar/model.py`, plus the generic decode path from M6 in [`sidecar-implementation-plan.md`](sidecar-implementation-plan.md):

- `ResizeSpec`: add `stretch`, `pad_position`, `rounding` and array `pad_value`.
- `InputSpec`: add `color_format`.
- `CropPolicySpec`: add `pad_value`.
- `OutputSemantic`: add the new semantics. Output dtypes gain `int32`/`int64`/`bool`. `OutputSpec` gains `class_axis` and `packed_keypoints`.
- `DetectionDecodeSpec`: add `selection`, `top_k`, `nms`, bindings, `box_scale`, `score_activation`, `class_id_map`, `clamp_boxes` and `is_person_only`. Remove the implicit `max_detections=300` at vNEXT.
- `SizeSpec`: restrict to `input.shape`, `resize.target_size` and `onnx`.
- Make `schema_version` visible to validators (validation context or a pre-validation branch) so that legacy sidecars (YOLOX, RTMPose) keep loading unchanged.
- Decoders: stretch/letterbox unprojection, `top_k` selection, host NMS stage order, `packed_keypoints` slicing, and pose-row gather by surviving detection index.

## Decisions

- **Decided — RTMDet RGB-backbone sizes** form their own family (name TBD; placeholder `rtmdet-rgb`).
- **Recommended — RTMDet resize is letterbox** (`top_left`, `half_up`, pad 114), derived from the mmdet `test_pipeline` that both benchmarks already use. Stretch measured −1.1 AP on Tiny.
- **Recommended — RTMDet NMS on the host** (`selection: nms`), with graph caps removed by the `codebase_config` fix.
- **Open — scope trims.** Accept the deferrals in S1, S3, S4, S5 and S7: `antialias`, `size_divisor`, `border_mode`, `softmax`, `class_id_map_before_selection`, `pre_nms_top_k`/`pre_nms_top_k_scope`, `min_bbox_size`, `graph_candidate_limit` and `instance_count`. Each can be added later as a non-breaking schema bump when a model needs it.

## Verification

1. Hand-write the five reference sidecars (S10) and load each through `load_sidecar()` under the vNEXT model.
2. Keep the existing YOLOX/RTMPose sidecars loading unchanged under the legacy path (`pytest skellytracker/tests`).
3. For each exporter, regenerate the sidecar and validate it with skellytracker's `SidecarModel` in that repo's tests.
4. Parity on COCO val2017 with sidecar-driven decoders:
   - RF-DETR matches `PostProcess`;
   - RTMDet meets the gate in step 6 of [Recommended changes](#recommended-changes) against a corrected PyTorch baseline;
   - YOLO26 det/pose match Ultralytics `val`.
