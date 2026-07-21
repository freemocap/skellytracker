# ONNX Batching and CoreML Compatibility

## Background

Downloaded YOLOX (and similar) ONNX checkpoints are exported with a **static batch dimension of 1** — every tensor shape has a literal `1` in the first axis. This is fine for a single camera, but for multi-camera setups you want to feed all frames to the model in a single call rather than looping over cameras serially.

The execution provider determines whether and how the batch dimension is made flexible.

---

## GPU / CPU: Dynamic Batch Surgery

For CUDA, TRT, DirectML, and CPU providers, `OnnxSession.create()` calls each model spec's `prepare` callback before loading the ONNX. For YOLOX this is `ensure_dynamic_batch` in `_yolox_dynamic_batch.py`, which rewrites the graph in two passes:

**Pass 1 — trivial rewrites**
- Input and output value_info: `dim_value=1` → `dim_param="N"` (symbolic)
- `Reshape` nodes whose target shape is fully static (e.g. `[1, 12, 320, 320]`): change `arr[0]` from `1` to `-1` (let the runtime infer batch from input)

**Pass 2 — dynamic shape computation**
- `Reshape` nodes whose target already contains a `-1` in a non-batch slot (e.g. `[1, -1, 4]`) can't simply get another `-1`. Instead, a small ONNX subgraph is inserted:
  ```
  Shape(data)         → [N, D1, D2, ...]
  Gather([N,...], 0)  → N  (scalar)
  Constant(tail)      → [arr[1], arr[2], ...]
  Concat([N, tail])   → [N, arr[1], arr[2], ...]
  ```
  The `Reshape` node is patched to consume this computed target instead of the hard-coded initializer.

**NMS bypass (YOLOX-specific)**
The rtmlib YOLOX checkpoints bake in a `NonMaxSuppression` subgraph containing a `Squeeze(axis=0)` node that only works for `batch=1`. The surgery adds `Identity` outputs that tap the raw pre-NMS tensors, so a caller can run Python NMS instead when `batch > 1`.

The result is a sibling `.dynbatch_v2.onnx` file (cached; regenerated if the source model is newer). Once loaded, you can pass a `(N, 3, H, W)` tensor and get N results back.

---

## CoreML: Batch=1, CPU Fallback

CoreML compiles the ONNX to a `.mlmodelc` ahead of time using Metal. Two things can go wrong:

1. **Dynamic batch dimensions** — a symbolic `"N"` in any tensor shape causes a compile failure. `OnnxSession.create()` skips the `prepare` callback for CoreML so the original static-batch model is used instead.

2. **Unsupported ops** — CoreML covers most common ops but not everything. The YOLOX NMS subgraph (a `Squeeze(axis=0)` feeding `TopK`) is not supported, causing CoreML to fail even on the unmodified model. The `GetCapability` log line reports how many nodes CoreML accepted before the failure.

`OnnxSession.create()` handles both cases with a catch-and-retry: if CoreML throws at model load time, it logs a warning and reloads the model with `provider="cpu"` instead. This is automatic and transparent to callers.

### What full CoreML support for YOLOX would require

Extracting the pre-NMS backbone (already possible via `ensure_prenms_model`) and running Python NMS after CoreML inference, matching what the GPU path does for batch>1. This would also allow CoreML to run at higher batch sizes via a static batch rewrite. Not yet implemented.

---

## Summary

| Provider | `prepare` called? | Batch size | Notes |
|----------|------------------|------------|-------|
| `cuda` / `trt` / `directml` / `cpu` | Yes | Dynamic (any N) | Dynamic surgery enables variable batch size |
| `coreml` | No | Always 1 | Static model; multi-camera batching not yet supported |
