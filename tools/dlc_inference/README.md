# Independent DLC inference tools

This optional tool project reconstructs the supported inference model, loads its
FP32 checkpoint tensors with `weights_only=True`, and exports an ONNX model with
the existing Tracker sidecar format. It does not import DLC, DLC-live or
SkellyClicker. Training/reference capture remains in `../dlc_training`.

Use this directory's included `uv.lock` and Python version. Set an explicit
`UV_PROJECT_ENVIRONMENT` outside existing Core/training environments and run
`uv sync --frozen --project PATH_TO_THIS_DIRECTORY`. The environment includes
Torch/timm only for conversion; a deployed Tracker uses its existing ONNX Runtime.
No installation of these dependencies in Core is required for consuming a bundle.

The supported model contract and reference tolerances are in
`../dlc_training/INFERENCE_CONTRACT.md`. Unsupported backbone/head/transform or
checkpoint fields fail explicitly. The inference model uses dense tensors and a
vectorized decoder; no per-keypoint Python loop runs in inference.

Run with the tool environment's Python:

```text
check_reference_parity.py CONFIG CHECKPOINT RAW_REFERENCE_DIRECTORY REPORT.json
export_model.py CONFIG CHECKPOINT BUNDLE_DIRECTORY --height HEIGHT --width WIDTH
check_onnx_parity.py BUNDLE_DIRECTORY/pose_model.onnx RAW_REFERENCE_DIRECTORY REPORT.json
```

Export creates a new directory and never overwrites an existing bundle. Spatial
dimensions are explicit and fixed; batch size is dynamic. No implicit resize or
padding is inserted. A different image/crop size requires an explicitly exported
size. Export shape constraints are not a claim that every batch size has been
benchmarked; the parity report names the batches actually tested. Use the registered `dlc` detector with the exported `dlc_resnet50_gn.yaml` sidecar.

The parity commands default to CPU. GPU validation requires the coordinator's
resource allocation and explicit provider selection. Reports measure numerical
equivalence, not tracking accuracy or real-time frame rate. CPU references and
the initial upstream GPU reference show a small cross-provider confidence
difference. Characterization established strict CUDA FP32 with `use_tf32=0` as
the supported contract; TF32 enabled failed the confidence tolerance and is not
certified. Do not silently widen tolerances to accept it.

Initial verification: raw independent CPU tensors and decoded outputs are
bit-identical to official DLC on four captured cases; ONNX CPU parity passed full
image batches 1/2/3, reversed ordering and a non-square offset crop. The generic coordinate decoder and registered detector also pass CPU tests through the Tracker factory, shared-session pool and DetectionStage. CUDA parity passed the same full/crop cases with `use_tf32=0`, with actual
CUDA convolution execution verified by profiling and fallback disabled. The
detector/session integration must enforce the recorded precision requirement.


## Standalone Tracker usage

Use `DlcDetectorConfig(model_name="/absolute/path/dlc_resnet50_gn.yaml")`
in `DetectionStageConfig(keypoint_detectors=[...], name="animal_pose")`.
Pass `onnx_overrides={"execution_provider": "cpu", "fp16": False}` to
`build_tracker` or `tracker_session_requests`. CUDA requires explicit `"cuda"`
and the artifact's `cuda_use_tf32: false`; automatic provider selection,
TensorRT, FP16 and unverified externally constructed sessions are rejected.
Other detector defaults remain unchanged; mixed sessions must explicitly choose
compatible settings. Sidecar model identity includes its absolute path.

The initial contract accepts native image dimensions only. Export a bundle for
each required image/crop size; arbitrary webcam dimensions are not automatically
resized. Tracker receives uint8 BGR images and normalizes to ImageNet RGB.
Keypoint order comes from the sidecar, not an anatomical mapping. Scores use the
existing inclusive threshold: a finite score equal to the threshold is accepted,
including zero when the user explicitly chooses threshold zero. Nonfinite rows
become all-NaN coordinates with zero confidence. No temporal filtering is added.

For a prepared reference artifact directory, run using a Tracker environment:

```text
check_tracker_runtime.py --artifact-root ARTIFACT_ROOT --report REPORT.json
```

This CPU check covers public config roundtrip/factory/pool construction, reused
sessions, partial/reordered batches, native-size crop inference and one outer
crop translation through DetectionStage. Its timing fields are diagnostic single
runs, not steady-state benchmarks. Raw ONNX CUDA parity reports establish
provider execution and numerical accuracy, not end-to-end tracking latency.


## Bounded CPU timing

`benchmark_tracker_cpu.py --artifact-root ARTIFACT_ROOT --report REPORT.json`
runs two warmups and ten timed repetitions per case, with two ORT CPU threads.
It reports p50/p95 preprocessing, batch inference, decoding and their combined
detector duration. It also verifies previously returned observation arrays are
unchanged after subsequent predictions. This is not an application FPS test.
The export declares a dynamic batch range of 1–32; numerical/runtime validation
covers batch sizes 1, 2 and 3, not exhaustive validation of all 32 sizes.

Foundation-001 CPU detector total p50/p95 milliseconds: 480×640 batch1
562.42/658.40; batch2 1356.62/1544.19; batch3 2165.38/2315.07; 227×391 crop
batch1 212.16/249.28. These are this machine's bounded run, not a throughput
promise. Strict CUDA parity is certified separately; no warm GPU latency
benchmark or live camera/UI validation was performed in this package.
