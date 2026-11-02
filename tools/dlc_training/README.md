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

## Delivered inference and synthetic foundations

W5A's independent ResNet-50 GroupNorm reconstruction, ONNX export, generic
coordinate decoder and registered Tracker detector are implemented in
`../dlc_inference` and Tracker runtime. Strict CPU/CUDA FP32 parity passed;
TF32-enabled CUDA is not certified. Native image sizes are explicit.

Forge's optional synthetic renderer has delivered the approved 15 fixed Benchy
features and a 160-image geometry dataset: 95 training, 30 evaluation, and 35
excluded guard-band views. Feature selection is authored, not generic automatic
feature discovery. Training and evaluation are separated by elevation. Synthetic
geometry does not establish real-camera identifiability or tracking accuracy.

## Prepare synthetic training

`synthetic_dataset.py` consumes files by path, with no Forge imports. It verifies
image/label hashes and exact split membership, preserves geometry feature order,
and leaves original coordinates, physical visibility and projection validity in
copied geometry artifacts. The DLC-only HDF boundary requires pandas MultiIndex;
it is an offline adapter, not a runtime dependency or frame-processing design.

The renderer uses image-edge origin and half-integer pixel centres. DLC 3.0.2
uses Albumentations xy keypoint transforms; installed Albumentations 1.4.3 flips
x to `(width - 1) - x` and rotates around `(width/2 - 0.5, height/2 - 0.5)`.
Therefore the adapter subtracts 0.5 from both projected components. A test checks
the actual installed flip operation. [@albumentationsRasterCoordinates143]
DLC's loader accepts strict
`0 < x < width`, `0 < y < height`; invalid labels become NaN only in the DLC
boundary. Physical occlusion does not remove finite valid labels. [@dlcSyntheticLabelLoader302] In dataset-v1,
441 occluded training labels remain and no valid geometry label is additionally
lost to the strict boundary. The one out-of-frame evaluation target is masked.

Run using the isolated DLC environment, with an existing cache and new run path:

```text
train_synthetic.py prepare RUN_DIRECTORY --source GEOMETRY_DATASET_DIRECTORY --hf-cache EXISTING_HF_CACHE
```

Preparation is CPU-only and offline. It copies only the 125 selected RGBs into
DLC's labeled-data group, explicitly passes the source split to DLC, and verifies
it again with `DLCLoader.load_split`. The `video_sets` synthetic.mp4 key is DLC's
labeled-image group identifier; no video is created or opened. All 160 original
geometry labels remain in the separate copied archive for provenance.

## Proposed training run — not executed

The prepared baseline uses ResNet-50 GroupNorm, 15 keypoints, batch 4 and native
640×480 images. Horizontal flip, affine transforms, crop sampling, resizing and
added noise are disabled; port/starboard identities are never silently swapped.
Rendered appearance variation supplies the initial training variation.
Inference also uses native dimensions without resizing. AdamW uses the generated
learning rate 0.0005; its decay milestones 90/120 are beyond this proposed
10-epoch baseline. Evaluation interval is 10; snapshots save every epoch with
DLC's configured retention limit. No synthetic checkpoint or evaluation result
exists until training/evaluation run.

```text
train_synthetic.py train RUN_DIRECTORY --epochs 10 --device cuda:0 --hf-cache EXISTING_HF_CACHE
```

The command requires an allocated GPU slot. It verifies preparation/config/source
hashes, image hashes, and explicit device availability before training. It refuses
a run that already has a training execution report or checkpoint; use a new run
for a new experiment. `training_execution.json` records running/completed/failed
status, elapsed time and retained checkpoint hashes separately from the immutable
preparation manifest. Offline mode prevents a hidden weight download.

Resource planning is unmeasured: 95 images at batch 4 imply 24 batches per epoch,
about 240 batches for 10 epochs. Budget roughly 4–10 GiB VRAM and several minutes,
then measure the first epoch before committing to a longer slot; do not present
this as a benchmark or extrapolate foundation-example speed directly. Stop on
allocation failure rather than silently changing batch size or image geometry.
The cached ImageNet ResNet-50 GN weights support initialization, not a previously
trained Benchy model. B4/B5 remain incomplete pending actual training, evaluation,
conversion parity and failure-case delivery.


Seed 20261101 is stored in the generated training config and execution report;
the renderer's independent seed remains in the geometry manifest. GPU kernels
are not promised bitwise deterministic. Ctrl+C is recorded as failed with
elapsed time and already-saved checkpoint hashes; force-kill can leave running
status as evidence of interruption. No resume workflow is provided here. Failed
runs are preserved and retry requires a new explicit preparation directory.


## Initial synthetic evaluation result

The fixed terminal epoch 10 was trained and evaluated on the 30 held-out views.
DLC retained it as `snapshot-best-010.pt`; selection uses its explicit filename
and verified checkpoint epoch 10, not an arbitrary best-snapshot lookup.
`evaluate_synthetic.py RUN OUTPUT --snapshot snapshot-best-010.pt --hf-cache CACHE`
uses CPU only, preserves exact source/feature ordering, and compares predictions
to original geometry after adding 0.5 to restore edge-origin coordinates.
`--refresh-report` recomputes summaries and overlays from saved predictions
without repeating model inference.

This short baseline is poor: unfiltered mean Euclidean error is 77.23 pixels
across 449 valid targets; PCK within 20 pixels is 34.52%. Visible targets average
66.77 pixels and occluded targets 101.82 pixels. All 449 predictions are finite;
one projection-invalid target is excluded. A diagnostic confidence cutoff 0.6
reduces mean error to 22.60 pixels but includes only 10 of 449 targets (2.23%),
excluding 439. It must not be presented as representative tracking accuracy.
The report also includes true root-mean-squared Euclidean distance separately,
per-feature statistics, all raw confidence values, and ranked failure overlays.
This is synthetic held-out evidence only; it does not validate real cameras.

## Controlled training-budget comparison

For the approved 10-versus-100 epoch experiment, prepare a NEW run with
`--epochs 100 --save-epochs 10 --max-snapshots 12 --batch-size 4 --augmentation-policy diagnostic-none`, using the same geometry source
and offline weight cache. Preparation stores immutable project/pose configurations.
Training reads its save interval from that verified configuration and rejects an
epoch count different from the prepared count. Retaining 12 regular checkpoints
preserves epoch 10 through epoch 100, including DLC's promotion of a previous
best checkpoint back to a regular checkpoint. The actual DLC retention regression
exercises this behavior; default retention of five loses epoch 10.
[@dlcSnapshotRetention302]

Select comparison checkpoints by their saved epoch metadata and hashes, never by
best-score selection. This experiment changes the training budget and checkpoint
storage policy only; it does not change the data, features, image dimensions,
augmentation, batch size or configured learning-rate schedule. Its existing
learning-rate milestone at epoch 90 remains in effect. CPU analysis of the earlier
model showed 9.66-pixel mean error on training views versus 77.23 pixels on held-out
views; more training is an experiment, not an established generalization fix.
GPU training requires the coordinator's explicit slot release and assignment.

## Production preparation

Preparation defaults to the installed DLC training augmentation recipe, with
horizontal mirroring disabled because the fixed geometry points have no declared
mirror correspondence. DLC applies its affine transforms, crop sampling, noise
and motion blur and transforms the keypoints with the images. Inference
preprocessing remains separate. `--augmentation-policy diagnostic-none` is an
explicit diagnostic choice, not the production default. [@dlcPytorchConfiguration]

The configurable starting budget is 200 epochs with batch size eight, matching
DLC's training settings defaults. This is not a convergence guarantee or a GPU
allocation. Measure throughput and memory before starting the full run, retain
training/development curves and report actual optimizer updates.
[@dlcTrainingSettings302]

For production datasets, the training-time evaluation split is `development`.
The `final_evaluation` images are excluded from the DLC training project and
automatic diagnostic evaluation. Canonical reference images are outside all
learning splits. Source image indices remain explicit when images are copied
into DLC's flat labeled-image directory, avoiding duplicate filename collisions.
