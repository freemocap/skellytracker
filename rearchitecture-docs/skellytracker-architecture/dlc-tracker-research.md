# DeepLabCut tracker feasibility audit

**Superseded recommendation:** The owner subsequently specified no DeepLabCut, DLC-live, or SkellyClicker dependency, including a DLC worker or DLC-dependent conversion tool. The current recommendation and revised estimate are in [Independent inference for DLC-trained models](dlc-independent-inference.md). This document retains the original source audit; its DLC-runtime and worker proposals are historical and must not guide implementation.

Audit date: 2026-10-30. Source inspection and dependency metadata review; no model execution, package installation, or benchmark. Estimates below are engineering judgments. This note proposes work, not an implemented tracker.

## Assessment

A DLC tracker is feasible for both real-time and post hoc FreeMoCap processing. SkellyTracker already separates detector logic from backend resource ownership; ONNX is one implementation, not its fundamental contract. Add a persistent PyTorch DLC backend and a keypoint adapter. Use the same inference implementation for incoming images and decoded video frames.

The significant work is dependency isolation, safe multi-camera execution, and letting FreeMoCap consume project-defined keypoint schemas. Producing custom 2D keypoints is a smaller feature than supporting arbitrary animals or objects throughout 3D reconstruction, biomechanical fitting, recording, and visualization.

Recommended first scope: one specified DLC3 PyTorch project, one subject per image, fixed/full-frame input, explicit snapshot selection, preserved confidence, and standalone SkellyTracker inference. Follow with efficient camera batching and FreeMoCap integration. Support additional model families and multi-animal identity only after this contract is proven.

## Inspected checkout state

| Repository | Inspected state |
| --- | --- |
| SkellyTracker | `sfn26`, HEAD `84fb000d067eecbabde49cca37f2b687850890a9`; existing untracked citation tools and STL research note preserved. |
| FreeMoCap core | `sfn26`, HEAD `34d590750bd2548789f2f0b847803db08cd43828`; substantial existing working-tree edits, including recording and post hoc paths. Findings describe the inspected working tree. |
| SkellyClicker | `../other/skellyclicker` relative to workspace root; HEAD `9481ed19ce85a90cd8ff50909be8080fbf41afbd`, `.git/HEAD` says `main`. Git status failed under available access, so cleanliness was not established. Files were readable. |

Core currently declares editable sibling sources in `[tool.uv.sources]`, with `sfn26` in separate workspace clone metadata. Its lockfile and installed SkellyTracker `direct_url.json` also point at the local checkout. This differs from the older workspace orientation; no source declarations, environments, or Git state were changed. These existing path declarations do not authorize introducing more path overrides.

The host reports an NVIDIA RTX 3090 with 24 GiB VRAM and driver 610.60. No trained checkpoint was found by the targeted search in the SkellyClicker checkout. Core's environment has NumPy 2.4.6; no installed DLC/PyTorch distribution was found by the targeted directory inspection. This is not an inventory of all models or environments on the machine.

## What SkellyClicker provides

Relevant files, relative to the SkellyClicker root:

- `skellyclicker/core/deeplabcut_handler/deeplabcut_handler.py`: reads project bodyparts, skeleton connections, and iteration; calls its video analysis wrapper; optionally filters predictions and exports labels.
- `skellyclicker/core/deeplabcut_handler/analyze_videos_dlc.py`: resolves the PyTorch model folder, inference configuration, pose snapshot, optional top-down detector snapshot, and creates persistent DLC pose/detector runners before analyzing videos.
- `skellyclicker/core/deeplabcut_handler/create_deeplabcut/deelabcut_project_config.py`: training defaults include `resnet_50`.
- `pyproject.toml`: pins DLC `3.0.0rc13`, NumPy `1.26.2`, and pandas `2.1.4`.

This is substantive precedent: DLC inference already operates on a loaded runner beneath its video-file API. The useful reusable knowledge is project/snapshot resolution and DLC preprocessing/postprocessing semantics. SkellyTracker should depend on a selected DLC runtime, not import SkellyClicker.

Do not transplant the whole workflow. It defaults to a process per video and passes runner objects into those jobs, writes pickle/HDF/CSV output, and calls offline trajectory utilities. Its handler uses batch size 8 with a comment that 16 is too large for five mocap videos; this is anecdotal, not a benchmark. Its merged annotation CSV explicitly drops likelihood columns. A tracker adapter must retain those likelihoods. The default median filter is also not a live inference requirement.

## Available upstream routes

| Route | Input and execution | Assessment |
| --- | --- | --- |
| Native DLC3 PyTorch runner | Resolve `config.yaml`, model configuration and snapshot; retain runner and call it on image arrays or batches | Best fit for direct project-path input and a shared implementation for live/post hoc. Requires a version-specific compatibility layer. |
| DeepLabCut-live PyTorch | Export a model bundle, load `DLCLive`, initialize once, call `get_pose` on frames | Official purpose-built live API. Useful alternative, with its own dependency and per-stream state constraints. |
| DLC video analysis plus result import | Analyze files and import original per-frame coordinates and confidence | Useful post hoc reference and possible transitional feature; does not itself supply live inference. |
| ONNX conversion | Export the underlying model and reproduce the relevant surrounding operations | Possible engineering investigation for selected models; not a prerequisite or a universal project loader. |

Native DLC v3.0.2 exposes `get_pose_inference_runner` and `get_detector_inference_runner` in `pose_estimation_pytorch/apis/utils.py`. Its video API constructs these once and passes frames into inference. The runner accepts image iterables and batches; initialization loads weights and selects evaluation mode. Its mutable batch/context/result state means concurrent use needs explicit serialization. Initially disable its optional internal preprocessing thread for short per-call image batches and measure before enabling it. [@dlc302Utils; @dlc302Videos; @dlc302Runner]

Current DeepLabCut-live source supports PyTorch and legacy TensorFlow routes. PyTorch can return either `(keypoints, 3)` or `(detections, keypoints, 3)` arrays. The third component is confidence, not depth. Dynamic cropping and previous-pose state belong to an individual image stream. Its `model_path` expects an exported artifact; pointing it at a training project directory is not equivalent. The inspected main metadata identifies `1.1.1rc1`; verify the chosen released artifact separately before implementation. [@dlcLiveApi20261030; @dlcLiveDependencies20261030]

DLC v3.0.2's PyTorch `export_model` writes a `.pt` bundle with model configuration, pose weights, and optionally detector weights. It does not perform ONNX conversion. Preserve this distinction: exporting for DLC-live is already supported, while deploying a model through ONNX Runtime would require its own correctness work. [@dlc302Export]

The inspected DLC-live API lists `pytorch` separately from TensorFlow `base`, `tensorrt`, and `lite` model types. Do not assume its TensorRT setting accelerates a PyTorch model. For an ONNX experiment, validate model architecture/export support, heatmap decoding, preprocessing, crop-coordinate inversion, and optional detector/assembly logic against native DLC outputs. A successfully exported backbone alone does not demonstrate equivalent tracking. [@dlcLiveApi20261030; @dlc302Export]

PyTorch does provide an ONNX exporter; it is inaccurate to call conversion categorically impossible. That does not establish exportability or parity for an arbitrary DLC project. Native inference remains the lower-risk first integration. [@pytorchOnnx215]

## Dependencies are an immediate implementation decision

| Constraint | Current local/upstream declarations | Consequence |
| --- | --- | --- |
| NumPy | SkellyTracker `==2.4.6`; core override `>=2.3.4,<2.5`; inspected DLC-live main `>=1.20,<2` | DLC-live cannot be added unchanged to the existing environment. |
| File locking | SkellyTracker `filelock>=4.0.8`; DLC v3.0.2 `filelock>=3.12,<3.16` | Full DLC cannot be added unchanged either, despite its compatible base NumPy range. |
| OpenCV | SkellyTracker uses `opencv-contrib-python`; both inspected DLC distributions request `opencv-python-headless` | Review duplicate `cv2` ownership during packaging. Tracker currently excludes `opencv-python`, not the headless distribution. |
| Python | Tracker `>=3.11,<3.13`; full DLC `>=3.10`; DLC-live `>=3.10,<3.13` | A shared supported Python version exists; it does not resolve the other package constraints. |
| GPU stack | Existing tracker uses ONNX Runtime CUDA packages; native DLC adds Torch/torchvision | Select compatible runtime wheels and test device selection, memory use, startup and coexistence. |

Upstream dependency claims above come from version-pinned full DLC metadata and the dated DLC-live main source. Full DLC v3.0.2 base dependencies no longer cap NumPy below 2; some TensorFlow extras still constrain it. Do not generalize the DLC-live NumPy conflict to every DLC3 runtime. [@dlc302Dependencies; @dlcLiveDependencies20261030]

Recommend a persistent DLC worker in its own environment for the first implementation, while separately investigating whether supported dependency updates can make an in-process installation practical. The worker should load the selected model once, accept bounded frame batches, and return coordinates/confidence/metadata. A small host-side `DlcSession` proxy can retain the existing SkellyTracker API. The worker should not import the host SkellyTracker package if that reintroduces its conflicting dependency requirements.

Use explicit model paths and a versioned protocol; preserve camera IDs and frame timestamps. Start with a measured binary transport, then use shared-memory image buffers if copies materially affect latency. Four 1920x1080 BGR images at 30 Hz represent about 746 MB/s of raw image payload before extra copies, so pickling full-resolution images casually is a poor default. Define ownership until inference completes, cancellation, worker failure reporting, and cleanup.

An alternative is an intentionally supported in-process package set after dependency review and regression validation. Simply overriding incompatible requirements, downgrading core NumPy, or using `--no-deps` is not evidence that the runtime works. No such change was attempted in this audit.

## Fit within SkellyTracker

The following are local source findings, with paths relative to this repository:

| Existing seam | Source | DLC implication |
| --- | --- | --- |
| Backend-neutral lifecycle | `skellytracker/core/sessions/session.py` | Implement `create`/`close` and a distinct `kind`; ONNX conversion is unnecessary. |
| Detector interface and registry | `skellytracker/core/detectors/detector_base_classes.py` | Add a DLC config and keypoint detector with explicit preprocessing and output conversion. |
| Shared resources, independent trackers | `skellytracker/core/sessions/shared_sessions.py` | Share compatible model resources; never accidentally share camera/pipeline temporal state. |
| Single-image and camera-batch input | `skellytracker/core/tracker/tracker.py` | Existing live and video callers can feed the same detector. |
| ONNX-specific batching | `skellytracker/core/tracker/detection_stage.py:590` | Generalize the batching capability or add a deliberate DLC dispatch path. |
| Non-ONNX fallback | `skellytracker/core/tracker/detection_stage.py:630` | Currently clones per-camera detectors and invokes them through a thread pool with a shared session. A shared mutable DLC runner would race. |
| Named keypoints and confidence | `skellytracker/core/data_primitives/keypoints.py` | Adapt DLC bodyparts in model order; preserve confidence and image coordinates. |
| Configurable annotation edges | `skellytracker/core/annotation/keypoint_annotator.py` | Project skeleton edges can configure overlays without requiring a standard human. |
| Recorded-video processing | `skellytracker/core/io/process_video.py` | Already runs framewise or camera-batched tracking and resets temporal state between recordings. |

Proposed components:

1. A project resolver producing an immutable descriptor: normalized config path, engine, iteration, shuffle, training fraction, concrete snapshot and optional detector snapshot, ordered bodyparts, display edges, preprocessing settings, versions, and content fingerprints. Resolve “latest” once at startup and record the selected artifact. Validate moved-project paths without silently rewriting training data.
2. A `DlcSession` that owns the resident runner or worker and its GPU lifecycle. Session reuse keys must include model identity/content, device, precision and inference options. Same path with newly trained weights must not reuse stale resources.
3. A `DlcKeypointDetector` that maps BGR input and DLC outputs into `Keypoints`/`Observation`. Retain DLC's own model-specific transforms and decoder. Convert colors exactly once; reverse crop/resize transforms exactly once.
4. A capability-based camera-batch interface, instead of treating `kind == "onnx"` as synonymous with batch support. A serialized fallback is acceptable for correctness; measure before claiming multi-camera performance. Keep mutable per-stream crop/tracking state separate from shareable model weights.
5. Optional runtime installation and lazy imports, so ordinary SkellyTracker use does not require importing DLC or loading Torch.

Important output detail: local image-space detectors use `xyz[:, :2]` for pixel coordinates and a zero z placeholder for valid 2D detections; low-confidence rows can be NaN. `Keypoints.valid_mask` checks all three components, so setting z to NaN for every valid DLC point would invalidate it. DLC likelihood belongs in `visibility`, never in z. The placeholder is not a measured 3D coordinate. Confidence is a quality gate, not a calibrated cross-model probability.

Begin without dynamic cropping. SkellyTracker already manages crops; enabling both its crop policy and DLC's crop history requires careful composition. For optimization, use one explicit crop owner and independent state per camera. Full-frame single-subject DLC needs no YOLOX person detector. Top-down models may require their own detector; the existing human detector is not an interchangeable animal/object detector. The inspected native runner requires batch size one for dynamic cropping and CTD tracking; use a stateless inference configuration for initial cross-camera batching. [@dlc302Runner]

## Fit within FreeMoCap

Relevant paths below are relative to the core checkout (`../freemocap` from this repository):

- `freemocap/core/pipeline/inference_service.py`: backend-neutral registrations, shared session pool, one managed inference execution thread, one outstanding request per client, round-robin scheduling with a post hoc turn after three ready real-time turns. Requests are not combined across clients.
- `freemocap/core/tracking/tracker_factory.py`: explicit backend allowlist `cpu`, `mediapipe`, `onnx`; must gain DLC session requests and configuration.
- `freemocap/core/pipeline/realtime/realtime_skeleton_inference_node.py`: chooses RTMPose or MediaPipe, drains stale live requests, reads matching camera frames, submits a camera batch.
- `freemocap/core/pipeline/dataflow/tracker_adapter.py`: submits decoded recording batches through the same service with original recording timing and bounded read-ahead.
- `freemocap/core/pipeline/posthoc/mocap_detection.py`: another inspected post hoc coordinator using the shared service; active recording paths are under local development.
- `freemocap/core/tasks/mocap/mocap_task_config.py`, realtime camera configuration, and UI `mocap-slice.ts`/`mocap-detector-settings.tsx`: restrict choices to current tracker types.
- `freemocap/core/tasks/mocap/tracker_mappings.py`: hardcoded tracker schemas and standard-human mappings.
- `freemocap/core/tasks/mocap/posthoc_mocap_task.py`: builds a standard-human bundle and assigns detector-specific recording provenance.
- `freemocap/core/skeletons/standard_human_skeleton.py`: loads the human model and its landmark mapping.

The existing inference service can remain the scheduling owner. It should call the new backend through the existing tracker contract, including when that backend proxies to an isolated worker. This avoids inventing a separate live pipeline. However, scheduling is non-preemptive: a long post hoc DLC request or model setup can still stall live work. Warm up before live readiness and bound offline batch duration; a 3:1 scheduling policy is not a latency guarantee.

Three distinct deliverables should remain visible in scope:

1. **Custom 2D tracking:** arbitrary ordered point names, confidence, overlays, recording and exports. Broadly supported by the detector primitives; core's configuration and schema handling need changes.
2. **Custom 3D points:** calibrated, synchronized views of corresponding physical landmarks, with consistent point identity across cameras. Geometry can consume named observations, but the current task setup must permit this without forcing a standard-human bundle.
3. **Custom fitted skeleton/biomechanics:** authored landmark mappings and a suitable model, with geometry/scale and semantics for those particular points. A DLC project's display edges alone do not specify rest geometry, segment frames, joint axes or mass properties. Human skin markers, eye contours, ferret landmarks and rigid-object markers cannot share human fitting assumptions automatically.

A DLC model can also be a supplemental stage, such as eyes or custom markers alongside RTMPose. SkellyTracker stages make this plausible, but core's present detector selection is not a general multi-tracker UI. Treat composition, namespacing, model reuse and additional GPU load as a follow-on feature.

Multi-animal support is another step. SkellyTracker has `MultiPersonTracker`, but the inspected shared inference service leases ordinary `Tracker` objects and expects one `Observation` per camera. DLC's detection dimension cannot simply be flattened away. It needs explicit subject identities, temporal association, cross-camera association, and compatible downstream recording/reconstruction. Offline tracklet stitching is not an online identity guarantee.

## Performance expectations and measurement

Real-time inference is technically plausible on the available RTX 3090, especially for one subject and controlled crop resolution. No honest model-specific FPS estimate follows from the GPU name, project path, or number of landmarks alone. Backbone, input size, detector, CPU preprocessing, camera count, GPU contention and stateful reacquisition determine latency. Full-resolution ResNet inference may be much more expensive than an existing pose model operating on a compact crop.

For historical context, the official dog-video benchmark reports Windows GTX 1080 inference at 300x533: MobileNetV2-0.35 94 +/- 7 images/s and ResNet-50 49 +/- 1 images/s. At 600x1067 those become 41 +/- 1 and 18 +/- 0 images/s. These older benchmark configurations illustrate resolution/backbone sensitivity; they do not predict current PyTorch performance on the RTX 3090 or end-to-end FreeMoCap latency. [@dlcHistoricalSpeed]

For C cameras at F synchronized frame groups per second, required throughput is C x F images/second. If images execute serially with mean service time t milliseconds, the optimistic group-rate ceiling is 1000/(C x t), excluding capture, transfer, reconstruction, rendering and recording. These are arithmetic examples, not DLC measurements:

| Hypothetical service time per image | 1 camera ceiling | 4-camera group ceiling |
| --- | ---: | ---: |
| 5 ms | 200 Hz | 50 Hz |
| 10 ms | 100 Hz | 25 Hz |
| 20 ms | 50 Hz | 12.5 Hz |

Four cameras at 30 Hz require 120 images/s; serial inference must average below 8.3 ms/image before overhead. A true four-image batch measured at 20 ms would instead provide at most 50 groups/s before overhead. Camera batching can improve GPU utilization without waiting for additional temporal frames. Temporal batching raises live latency and may conflict with per-stream dynamic cropping.

Treat 30 Hz multi-camera operation as a benchmark target, not a delivery promise. Sustained live operation at a lower tracking rate is still possible because the current live adapter drops stale requests; it must keep synchronized frame identities. Post hoc mode should process every required frame, even when slower than recording rate. At 100 aggregate images/s, four 30 Hz videos take approximately 1.2 times recording duration for inference alone.

Measure with one representative SkellyClicker-trained checkpoint and held-out frames from the intended camera views:

- Model load and warmup separately from steady state; first frame after model switching.
- Identical preprocessing and snapshot against DLC reference output; errors in pixels and confidence, including crop edges, missing detections, and full-frame coordinate restoration.
- 1, 2, 4 and intended maximum camera counts; normal input resolution and smaller valid crops; batches with missing/empty cameras.
- Aggregate images/s, synchronized group Hz, capture-to-result median/p95/p99 latency, maximum stalls, dropped live groups, memory/VRAM and GPU utilization. Synchronize GPU work correctly when timing.
- One-at-a-time execution versus true camera batching; FP32 baseline before FP16 or compilation. Compare accuracy as well as speed.
- Live and post hoc running together, rendering/recording enabled, loss/reacquisition, start/stop, cancellation, model switching, and long-run memory stability.
- CPU-only startup as a supported correctness fallback if desired; do not assume it meets multi-camera live targets.

No camera capture, model loading, dependency changes, or inference benchmark was performed for this audit. Literature demonstrations of DLC-live establish feasibility, not the performance of this particular project and pipeline.

## Effort and staged delivery

Planning estimates for one engineer familiar with these repositories, with a trained representative model supplied. They are not commitments; unknown model compatibility and packaging can expand them.

| Stage | Deliverable | Estimated focused effort |
| --- | --- | --- |
| Feasibility spike | Select runtime, load one checkpoint persistently, compare image outputs, measure 1/4-camera workloads and resolve deployment choice | 2-5 days |
| Standalone SkellyTracker backend | Project resolver, session/detector, isolated worker if needed, batching, state/reset, errors and meaningful regression checks | Additional 1-2 weeks |
| Core integration | Real-time and post hoc configuration, project selection, dynamic schema/provenance, overlays/exports and a validated reconstruction path | Additional 1-2 weeks for constrained scope |
| Broader support | Multiple DLC families/legacy TF, automatic export/cache, multi-animal identity, generalized skeleton workflows and polished distribution | Several additional weeks; specify separately |

A useful constrained end-to-end feature is roughly a 3-5 week effort with contingency for dependency isolation and unknown model compatibility. A single-image proof can be much quicker. Arbitrary DLC projects working automatically throughout FreeMoCap are a larger feature.

Repository boundaries: implement and validate standalone DLC support in SkellyTracker first, with no imports from SkellyClicker, SkellyCam, SkellyForge or FreeMoCap. Hand that change to the human for commit/push on the agreed branch before modifying core for integration. Core owns cross-project contracts, camera scheduling and mapping to reconstruction models. Any new SkellyForge model support has its own separate stage. No implementation or downstream dependency refresh is part of this audit.

Success for the first release means: select a supported project and explicit model, obtain matching named 2D observations from both live and recorded input, retain frame identity and confidence, isolate camera state, and report measured performance. Treat biomechanical fitting and multi-subject support as explicit capabilities rather than implying them from successful inference.
