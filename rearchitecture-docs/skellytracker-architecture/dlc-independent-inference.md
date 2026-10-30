# Independent inference for DLC-trained models

Date: 2026-10-30. Revised architecture following the owner's explicit dependency constraint. Code and documentation assessment only; no inference implementation, installation, export or benchmark was performed.

## Decision and feasibility

Build a SkellyTracker-owned importer for supported DLC-trained model artifacts. It reads the project configuration and trained weights as data, constructs an inference-only network, and produces a versioned SkellyTracker model bundle. Prefer an ONNX bundle so live and recorded frames use the existing `OnnxSession` and camera batching.

There must be no dependency on DeepLabCut, DeepLabCut-live or SkellyClicker in the importer, conversion tool or deployed tracker. A separate environment containing DLC does not meet this requirement. The earlier proposal to wrap a DLC runner is superseded.

This is feasible without recreating the DLC application. We need the supported network's forward computation, input transforms and output decoder. Training loops, loss functions, target generation, dataset management, GUI, analysis scripts, pandas/HDF output and DLC's scheduler are outside the inference requirement. DLC's own model builder constructs training components alongside inference components; importing that builder would preserve unwanted coupling. [@dlc302PoseModel]

Recommend first supporting the single-subject PyTorch ResNet/heatmap family used by SkellyClicker's defaults. Compatibility should be identified from the concrete backbone, head, decoder and transform configuration. The marketing label `resnet_50` alone does not define a complete inference contract.

## Evidence and boundaries of this assessment

Local SkellyClicker source pins `deeplabcut==3.0.0rc13`. Its `DeeplabcutTrainingConfig.model_type`, project creation defaults and example pipeline select `resnet_50`. Its analyzer explicitly chooses the PyTorch model folder, reads `pytorch_config.yaml` and `pose_cfg.yaml`, and constructs a pose runner plus an optional top-down detector. Those are source-level observations, not a claim that every model ever trained with SkellyClicker uses those defaults.

Local source locations, relative to `C:/Users/jonma/code-stuff/github/freemocap/other/skellyclicker`:

- `skellyclicker/core/deeplabcut_handler/create_deeplabcut/deelabcut_project_config.py`
- `skellyclicker/core/deeplabcut_handler/create_deeplabcut/create_deeplabcut_config.py`
- `skellyclicker/core/deeplabcut_handler/run_deeplabcut_pipeline.py`
- `skellyclicker/core/deeplabcut_handler/analyze_videos_dlc.py`

Detailed upstream architecture below was verified against version-tagged DLC **v3.0.2** source. Attempts to retrieve the equivalent `v3.0.0rc13` URLs were unsuccessful. Exact rc13 parity is therefore not established. The importer must inspect and validate configurations and checkpoint tensors, rather than silently substitute v3.0.2 defaults. No local trained project is needed to assess this design; representative model artifacts and reference results become necessary for implementation acceptance and measurement.

SkellyTracker is on `sfn26` at `84fb000d067eecbabde49cca37f2b687850890a9`. Existing untracked research/citation files were preserved. Core's inspected development configuration points at the sibling tracker checkout; its separate workspace branch metadata says `sfn26`. Neither source declarations nor Git state were modified.

## What actually needs to be rebuilt

### 1. Project and checkpoint importer

Read only the relevant configuration fields: model identity, concrete architecture, ordered bodyparts, display edges, evaluation transforms, decoder settings and explicit selected checkpoint. Permit either a project directory or an explicit model-config/checkpoint pair. Directory discovery is a convenience boundary; do not spread DLC path conventions through runtime code.

Resolve iteration, shuffle and training fraction when needed. Reject ambiguous or unsupported combinations with a specific explanation. Inspect the actual model configuration even when the root project file says `resnet_50`. Preserve original files and record hashes of the inputs and importer version. A changed snapshot produces a new bundle rather than silently reusing a cache entry.

DLC v3.0.2's loader reads a PyTorch checkpoint dictionary and loads `snapshot["model"]` with `load_state_dict`. That model entry contains weights and buffers; it is not an executable architecture definition. We can create compatible modules ourselves and map the tensor keys. DLC's exported `.pt` bundle is another format containing configuration, pose weights and possibly detector weights; support for that format can be added without invoking DLC's exporter. [@dlc302Snapshot; @dlc302Export]

Use PyTorch's checkpoint reader at import time instead of inventing a tensor/pickle parser. Start with `weights_only=True` and CPU loading. Older checkpoint metadata can contain NumPy scalar objects that need explicit compatibility handling even when the model tensors are ordinary state dictionaries. Do not silently fall back to unrestricted object loading or import arbitrary classes named in a file. Conversion should extract only the supported model data and write a separate artifact. [@pytorchSerialization215; @dlc302Snapshot]

Strictly account for model tensor keys, shapes and dtypes. Do not hide unsupported architecture differences behind `strict=False`; known non-inference extras need explicit handling. Tensor shape equality alone does not prove correct stride, padding, normalization or decoding.

### 2. Inference-only network

The verified v3.0.2 ResNet-50 configuration uses `resnet50_gn`, output stride 16, and 2048 feature channels. The backbone wrapper delegates to `timm.create_model`, removes the classifier and calls `forward_features`. GroupNorm and the output-stride configuration matter; a standard torchvision ResNet-50 is not a drop-in equivalent merely because its name contains ResNet-50. [@dlc302ResnetConfig; @dlc302ResnetBackbone]

The default bodypart head has two separate transposed-convolution branches: one produces K heatmaps and the other 2K location-refinement channels. The inspected default uses kernel size 3 and stride 2. The implementation uses default zero padding and does not apply a final ReLU in these one-layer branches. Combined with the default backbone, the nominal output stride is 8. Reproduce the actual layer geometry; output array dimensions need not be exactly input dimensions divided by eight. [@dlc302BodypartHeadConfig; @dlc302SimpleHead]

Two reasonable construction choices:

| Choice | Dependency scope | Tradeoff |
| --- | --- | --- |
| Use a pinned compatible `timm` backbone and our own small heads | Optional importer/export tooling only | Fastest route to the existing architecture; still independent of DLC. Verify transitive packages and version compatibility. |
| Implement the supported ResNet variant using PyTorch primitives | Optional importer/export tooling only | More code and equivalence testing; stronger ownership of module layout and fewer constructor dependencies. |

Neither choice requires adopting DLC's registries, configuration classes or inheritance hierarchy. An independent implementation should preserve source provenance; any copied/adapted source also retains its applicable notices. This assessment does not require replacing PyTorch or ONNX Runtime themselves.

### 3. Deterministic preprocessing

Implement the supported evaluation transforms with the existing OpenCV/NumPy stack: explicit BGR-to-RGB conversion, configured resize/padding/cropping, normalization, float32 NCHW layout and the inverse coordinate transform. Do not import the training augmentation framework to perform a small set of deterministic transforms. The inspected transform builder uses ImageNet mean/std when `normalize_images` is enabled; configuration determines whether normalization, scaling and padding apply. [@dlc302Transforms]

The upstream preprocessor records offsets and scales to restore original coordinates. It does not recolor ndarray inputs in its `LoadImage` operation; relying on a `color_mode` argument alone can therefore produce incorrect colors. Our camera-image boundary must define this explicitly. Match interpolation, rounding and padding, not just target dimensions. [@dlc302Preprocessor]

Reuse the scheduling structure of the RTMPose detector, not its preprocessing constants or SIMCC decoder. Arbitrarily forcing a new crop/resize policy changes the input distribution and may change model accuracy. In particular, padding changes the image seen by a GroupNorm network; it should not be assumed numerically invisible.

### 4. Heatmap decoder

The default verified head disables sigmoid, clips reported scores to [0,1], and uses a location-refinement factor of 7.2801. For the default stride-eight case, the decoder finds each heatmap's maximum at `(u,v)`, reads that point's two location-refinement values, and computes:

```text
x = (u + 0.5) * 8 + locref_x * 7.2801
y = (v + 0.5) * 8 + locref_y * 7.2801
confidence = clip(heatmap[v, u], 0, 1)
```

Then reverse preprocessing into image coordinates. These are the inspected defaults, not constants to impose on every model. The decoder must honor sigmoid, refinement and stride settings from the accepted configuration. Preserve location-channel ordering and argmax tie behavior. [@dlc302BodypartHeadConfig; @dlc302SinglePredictor]

This decoder is a bounded numerical implementation, not a reason to retain the DLC application. Initially return heatmaps and refinement maps from ONNX and decode with NumPy for inspection. Later, exporting the decoder into the graph could reduce device-to-host transfers; validate provider support and numerical equivalence before doing so.

### 5. Bundle, export and runtime adapter

Proposed imported artifact:

```text
model-bundle/
  model.onnx
  manifest.json
```

The manifest owns the point order, drawing edges, input/output tensor contracts, supported shapes, preprocessing/decoder settings, source hashes, converter version and provenance. It contains resolved data, not DLC Python objects or a mirror of the entire DLC project. Once imported, ordinary inference should work with the bundle alone.

Our own conversion command constructs our network, loads the trained weights and uses the PyTorch ONNX exporter. It must never call `deeplabcut.export_model` or use a converter that imports DLC transitively. Export is technically plausible for this convolution/normalization/deconvolution graph, but has not been executed in this audit. Operator lowering, target opset, GroupNorm handling and provider behavior are acceptance checks. [@pytorchOnnx215]

Dependency boundary:

- **Import/export:** our code plus explicitly selected Torch, optional timm, and ONNX export dependencies. No DLC packages or their utilities.
- **Tracking:** our detector, bundle, existing ONNX Runtime/OpenCV/NumPy stack. Torch/timm need not be installed on a machine that only runs a preconverted bundle.
- **Project selection UX:** import once, validate and warm up, then run the cached bundle. Changes to weights/configuration invalidate that cache. The source project remains untouched.

The optional converter can run in a separate tool environment if helpful, but it contains our implementation and selected generic ML tools, not DLC. Packaging the converter and its downloads is a real part of delivery; removing DLC does not make Torch's installation free.

## Fit to the existing architecture

Local SkellyTracker code already supports most of the runtime mechanics:

| Source | Existing capability | Required work |
| --- | --- | --- |
| `skellytracker/core/sessions/model_registry.py` | `ModelSource(local_path=...)` | Resolve the imported ONNX artifact. |
| `skellytracker/core/sessions/onnx_model_spec.py` | Named model descriptors and warmup input shape | Produce a descriptor from the bundle manifest. |
| `skellytracker/core/sessions/onnx_session.py` | Persistent models, provider selection, batching and cleanup | Validate exported tensor contracts and supported shapes. |
| `skellytracker/core/tracker/detection_stage.py` | Batched ONNX preprocessing/inference/postprocessing | Add the detector using this existing path. |
| `skellytracker/core/data_primitives/keypoints.py` | Named points and confidence | Preserve point order, missing-point semantics and full-image coordinates. |

An ONNX deployment removes the need for the new PyTorch session and generic non-ONNX batching changes proposed in the first audit. `run_batched` expects equal-shaped CHW images and outputs with a leading camera-batch dimension. Export a compatible batch axis and test partial camera batches; a fixed batch-one export will not satisfy that contract. Unequal image/crop shapes require an explicit supported resize policy or shape-grouped execution. Do not promise arbitrary dynamic-resolution batching from the existing `np.stack` implementation.

Core's `tracker_factory.py` currently recognizes only RTMPose/YOLOX when resolving ONNX model specs; extend that explicit selection to the imported-model detector. The same existing `InferenceService` can then schedule live and recorded images. Core still needs project/bundle selection, dynamic schema/provenance and downstream mapping support, as described in the original audit. No external runtime is necessary to deliver those changes.

Keep the responsibilities separate: SkellyTracker owns artifact compatibility and 2D inference. FreeMoCap owns camera coordination, recording, triangulation and adapters to skeleton models. A custom tracker does not automatically make an eye model or animal model a standard-human biomechanical model. Supplemental custom landmarks alongside RTMPose remain possible as a later stage-composition feature.

## What makes this bounded, and what expands it

| Support level | Assessment |
| --- | --- |
| One declared PyTorch ResNet/HeatmapHead profile, one subject per image | Bounded engineering work; recommended starting point. |
| Other supported ResNet depths or normalization variants with the same head | Incremental architecture profiles and tensor mappings, each tested. |
| HRNet, transformers, RTMPose/SimCC or other head types | Separate forward graphs and decoders; no automatic compatibility claim. |
| Top-down models with a separate detector | Detector import, crop geometry and association add scope. |
| Multi-animal bottom-up models | Multiple peaks, assembly and stable identities add substantial work. |
| Legacy TensorFlow checkpoints | Separate artifact reader, tensor-layout/name conversion and architecture fidelity; defer from the first implementation. |
| Every custom DLC project/version | Open-ended compatibility maintenance; not a useful first-release promise. |

Unknown profiles should fail explicitly. There must be no fallback that installs DLC or starts a DLC worker. Supporting the models our users actually train does not require implementing DLC's training or project-management systems.

## Correctness and performance

No retraining is inherently needed: reconstruct the same supported computation and load the existing weights. Correctness is the main engineering risk. A model can produce plausible points despite wrong color order, output stride, padding, score activation or a missing half-stride offset. Clipping scores before argmax can even change the selected location when several raw peaks exceed one; match operation order.

Validation plan:

1. Source-based tests for configuration resolution and strict tensor mapping; reject unsupported fields and ambiguous checkpoint selections.
2. Hand-constructed heatmaps/location maps for exact decoder coordinates, scores, ties and missing-data behavior.
3. Known image transforms for normalization, resize/padding and coordinate round trips, including odd sizes and edges.
4. Same-weights comparisons between our PyTorch graph and ONNX on batch one and multi-camera batches, inspecting intermediate/output tensors. This checks export fidelity, not independently DLC equivalence.
5. Compare against frozen reference predictions for supported trained models when available. These can be outputs already generated by SkellyClicker/upstream; no shipped or test-suite DLC dependency is required. Without such a reference, do not claim measured upstream parity.
6. Confirm the deployed tracker runs in an environment with no DLC, DLC-live, SkellyClicker, Torch or timm installed when using a converted bundle; converter checks separately prove no DLC imports/dependencies.

These future acceptance tests do not block a code-only feasibility assessment. No user model path is required for the present task.

Owning the execution removes unwanted package constraints and gives control over batching, memory, crops and exports. It does not reduce the trained network's intrinsic convolution cost. ONNX may improve execution, but no speedup multiplier is established. Benchmark supported models and provider settings before enabling precision changes.

Four cameras at 30 synchronized groups/s still require 120 images/s. Serial execution allows less than 8.3 ms/image before other work; true camera batches can improve throughput without waiting for future frames. Measure p95/p99 capture-to-result latency, batch time, dropped live groups, complete offline coverage, memory use and contention with concurrent post hoc work. Full-resolution inference and raw heatmap transfers may dominate. Import/export and model warmup belong outside steady-state tracking measurements.

## Revised effort estimate

Estimates are focused engineering time for one engineer familiar with this code, a bounded single-subject model profile and normal generic ML libraries. They are judgments from source inspection, not measured project plans.

| Milestone | Estimated effort |
| --- | --- |
| Minimal architecture/config reader, strict weights loader, independent forward pass and decoder | 3-7 days for the first profile |
| Converter, ONNX parity, bundle contract, SkellyTracker detector, batching and regression coverage | Additional 1-2 weeks |
| FreeMoCap project/bundle selection, live/post hoc wiring, custom schemas and constrained reconstruction | Additional 1-2 weeks |
| Packaging, model/version compatibility cases and integration hardening | Reserve about 1 week, potentially more if artifacts differ materially |

Budget approximately **4-7 weeks for a constrained end-to-end feature**, with a useful standalone prototype in the first week. Reimplementing the exact backbone in pure Torch rather than using a verified timm constructor adds roughly another week or two of architecture/parity work, depending on variants. Broad multi-animal, legacy TensorFlow and arbitrary architecture support can turn this into months of compatibility work.

The extra initial work replaces ongoing dependence on DLC with a small compatibility surface that we own. ONNX deployment also reuses the existing batch/session path, offsetting some of the cost of independent implementation. Scope the first release by a precise architecture profile and supported artifact versions rather than by a promise to support anything called a DLC model.

Implement standalone support in SkellyTracker first. Follow the human-owned commit/push handoff before core integration; any custom skeleton work in SkellyForge is a separate stage. This audit changes documentation and references only.
