# Isolated DLC training environment

This optional project provides the offline training environment for the DLC
integration experiment. It is separate from SkellyTracker's runtime dependencies.
DLC is permitted here for training and reference predictions; the planned model
converter and deployed detector must not import DLC or SkellyClicker.

Use Python 3.11 and this directory's included `uv.lock`. From this directory,
`uv sync --frozen` creates a local `.venv`; alternatively set
`UV_PROJECT_ENVIRONMENT` to an explicit isolated environment path. Do not install
these packages into Core's environment. The lock selects the CUDA 12.6 PyTorch
index; installed CUDA packages alone do not establish GPU availability or speed.

Run imports with `DLClight=True` and `MPLBACKEND=Agg` for headless use. Keep downloaded
datasets, checkpoints, predictions and reports in an ignored `.test-artifacts/`
directory. Training requires an allocated GPU slot when sharing this workstation.

The initial reference fixture is the official
`examples/openfield-Pranav-2018-10-30` example at DLC v3.0.2, commit
`4dcb9a7d5a701427ef78ecc8103d38eeef70601a`. A successful environment installation is
not a training, accuracy, conversion or inference-parity result. [@dlcFoundationExample302]

`fetch_example.py DESTINATION` downloads the pinned example and checks every file
against its upstream Git blob hash. Existing mismatched files cause an error.
`run_example.py prepare RUN_DIRECTORY --source DESTINATION` creates a separate
project and training split. Then run `train RUN_DIRECTORY --epochs 5` in an
allocated GPU slot, followed by `evaluate RUN_DIRECTORY` in a fresh process to
reload the saved checkpoint and save reference predictions. [@dlcFoundationApis302]

The short training run establishes a real checkpoint and reproducible reference
outputs for later conversion tests. It does not establish useful tracking quality
or synthetic-to-real generalization. Preserve the generated split and configs;
seeded runs alone do not guarantee identical GPU results across environments.

## Reference artifact contract

`reference_contract.py RUN_DIRECTORY` validates the saved artifact without
importing DLC or Torch. `keypoint_predictions.npz` contains `keypoint_values`, a
float32 tensor with axes `[image, subject, keypoint, component]`. The fixture's
components are `x`, `y`, `confidence`, in original-image pixels. This is a specific
DLC image-output contract, not a restriction on generic Tracker component counts.
Keypoint ordering comes from the trained model config; image ordering is recorded
in `prediction_metadata.json`. Subject axis zero is the example's single subject;
these independent example images do not establish temporal identity or timestamps.
Confidence is a model score, not physical visibility or an empirical measurement.
The JSON records the selected checkpoint path and SHA256. Preserve project and
PyTorch configs, train/test membership, source manifest and evaluation results.

## Next implementation branches (proposals, not implemented here)

**DLC inference branch — SkellyTracker:** independently reconstruct the supported
ResNet-50 GroupNorm model and heatmap/location-refinement head, load checkpoint
tensors, export ONNX, then add a registered detector using the existing factory
and session pool. Use the saved DLC predictions as the reference. The converter
and runtime must not import DLC, DLC-live or SkellyClicker. Begin with this exact
supported architecture and reject unsupported architectures explicitly.

Preserve the model's RGB conversion, ImageNet normalization, coordinate transform,
location-refinement scaling, confidence behavior and authored keypoint ordering.
Validate raw outputs and decoded original-image coordinates before claiming
parity. Include non-square images, transformed crops, batch tails and CPU/GPU
execution-provider comparisons. Define numerical acceptance tolerances in the
branch's checklist before implementation. [@dlcFoundationApis302]

The current `PoseDecodeSpec` schema accepts `heatmap` and `coordinate`, but
`simcc_pose_decode` explicitly rejects both. Schema acceptance is not runtime
support. Select and test the DLC decoder/export boundary in this branch; do not
feed DLC outputs to SimCC or overwrite shared interfaces owned by another task.

**Synthetic training branch:** establish a fixed, ordered set of meaningful
geometric features for the supplied object, with mesh hash, explicit source
units, object frame and reference scale. Point count follows the geometry.
Store object positions as `[keypoint, spatial_component]` and rendered image
targets as `[image, keypoint, image_component]` dense arrays with separate
visibility and validity arrays. Preserve projected labels for occluded features;
do not invent a model-confidence value for synthetic geometry. Store camera
intrinsics/extrinsics and each object pose so labels can be independently
reprojected. The STL alone must not silently establish units or feature identity.

Forge already owns `build_rigid_marker_skeleton`, which takes named reference-unit
positions and explicit origin/primary/secondary feature identities. Reuse it for
rigid-object reconstruction; do not implement another solver in Tracker or Core.
The synthetic image generator's repository assignment and the concrete supplied
object remain decisions for that branch's approval. Clicker is reference-only.

The two branches join when the synthetic-trained checkpoint passes the inference
branch's conversion/parity checks. Core then connects public Tracker and Forge
interfaces for live/posthoc use. Neither a demo checkpoint nor exact synthetic
geometry establishes real-camera STL tracking quality, symmetry resolution,
occlusion reliability or acceptable latency.
