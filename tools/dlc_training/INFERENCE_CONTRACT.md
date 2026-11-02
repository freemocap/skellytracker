# DLC inference contract: first supported family

This contract describes the approved W5A implementation target. It does not claim
that a DLC detector or ONNX conversion is already implemented.

## Supported input artifact

The initial artifact is foundation-001's `snapshot-005.pt`, SHA256
`cc500c01883d0cb01d0e48c7b65275f1d893bdcfa8f6eb591cc0d33265a40a86`, together with its
`pytorch_config.yaml`. The checkpoint contains `model`, `metadata`, and `scheduler`;
only `model` tensors define inference state. Safe `torch.load(weights_only=True,
map_location="cpu")` succeeds. Model loading must account for every tensor key,
shape and dtype and reject mismatches; no unrestricted pickle fallback.

The supported model is `ResNet/resnet50_gn`, output stride 16, no neck, and a single
`HeatmapHead` named `bodypart`. Its heatmap and location-refinement heads each have
one transposed convolution, kernel 3, stride 2, zero padding and no activation.
Heatmap output channels equal the ordered keypoint count K; location-refinement
channels equal 2K. No fixed keypoint count is imposed. The example has four.
The effective decoding stride is 8. Reject other architectures until explicitly
supported and tested. [@dlc302ResnetBackbone; @dlc302SimpleHead]

## Input and output semantics

Camera arrays enter Tracker as BGR uint8. This model consumes RGB float32 NCHW,
normalized with ImageNet mean/std. The example config applies no resizing,
padding, crop, flip or other inference augmentation. Preserve that behavior for
the reference; applying a different crop or resize is an explicit input operation
whose inverse transform must accompany each image. GroupNorm makes unrequested
padding a model-input change, not merely storage padding. [@dlc302Transforms]

The decoder selects the first row-major maximum per keypoint heatmap, adds the
corresponding x/y location refinement scaled by the declared 7.2801, and produces
model-input pixel centers using stride 8. Sigmoid is disabled and confidence is
clipped to [0,1]. Preserve raw heatmaps and offsets for parity testing before
exporting decoder arithmetic into ONNX. [@dlc302SinglePredictor]

Agreed shared boundary: one float32 output `[batch, keypoint, component]`, with
components `x`, `y`, `confidence` and native configured keypoint ordering. W4
confirmed there is no generic coordinate decoder and assigned its single reusable
implementation to W5A; W4's hand-specific output/alignment is not this interface.
The local and W4 `ImageTransform.restore_points` implementations agree: owned
float64 output, inverse scale/translation, no rotation. Decode to detector-input
crop coordinates; `DetectionStage` applies the outer crop translation exactly
once. Do not introduce a second generic decoder. Partial batches,
reordered camera submissions and arbitrary K must preserve image and keypoint
identity. Tracker's existing `Keypoints.visibility` carries backend confidence,
not physical visibility or calibrated probability; synthetic occlusion masks do
not replace it. Supply owned arrays to `Keypoints.with_values`, which does not copy.
Valid 2D values use z=0; missing coordinates use all-NaN xyz and confidence zero.
As in existing RTMPose detectors, scores below the configured threshold become
missing, with the threshold boundary inclusive. The new coordinate boundary must
explicitly reject malformed score ranges and mark non-finite predicted rows
missing; it must not publish infinities as valid keypoints. These decoder details
will be covered by tests in the detector implementation stage.

## Initial equivalence thresholds

Compare identical inputs and tensor meaning, not merely equal shapes:

- Normalized float32 input: `atol=1e-6, rtol=1e-6`.
- Raw FP32 heatmaps/location-refinement outputs: `atol=1e-5, rtol=1e-4`.
- Confidence on its [0,1] scale: absolute error at most `1e-4`.
- Decoded position: at most `0.01` model-input pixel per coordinate.
- Original-image position: propagate the input tolerance through the actual
  inverse-transform linear part, `abs(A) @ [0.01, 0.01]`, plus `1e-4` pixel.

Report maximum and p95 errors. Argmax ties and discontinuities remain visible
failures to investigate. These thresholds characterize conversion equivalence,
not tracking accuracy, and require CPU/CUDA verification before acceptance.
No FP16/INT8 variant is included in this first implementation. Official DLC's CPU
versus the previously saved GPU references differed by 0.000123084 confidence,
slightly above the provisional cross-provider threshold. Same-provider CPU
thresholds remain unchanged. Explicit CUDA characterization passed with
`use_tf32=0` (full/crop maximum confidence error 3.58e-7), while `use_tf32=1`
failed at 1.4424e-4. The supported initial CUDA contract therefore requires
float32 inputs and `use_tf32=0`; TF32 is not certified and no tolerance was relaxed.
The detector/session integration must carry this model-specific requirement into
provider options and shared-session identity, rather than changing every model's
precision policy or silently reusing a TF32-enabled session.

The optional DLC environment owns raw-reference capture only. The independent
converter and deployed Tracker must not import DLC, DLC-live or SkellyClicker.
The converter may use pinned Torch/timm; deployed inference uses the existing
ONNX Runtime infrastructure.
