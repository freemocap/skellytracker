# STL Tracker: mesh-based rigid object pose audit

Audit date: 2026-10-29. Documentation and source inspection only; no installation or performance measurements.

The requested capability exists under the names **model-based 6D object pose estimation**, **CAD-based pose estimation**, and **rigid object tracking**. Recommend the user-facing name **STL Tracker**, with a general mesh input and an explicit backend name. Ordinary calibrated RGB video is required. Depth is out of scope. Automatic acquisition and reacquisition are required; manual poses, point clicking and manually drawn boxes do not meet the user's workflow.

## Corrected recommendation: automatic color and silhouette matching

The user permits neon green/orange objects. Proposed workflow: automatically segment the distinctive color, retain the silhouette and bounding box, compare the silhouette against rendered views of the supplied mesh, then refine and report pose. Search again automatically after loss. This is an engineering proposal; its speed and robustness have not been measured.

STI-Pose specifically searches pose space by rendering the model and comparing silhouettes, without requiring object texture or annotated training poses. It accepts the object silhouette and model, so color segmentation could supply the silhouette. Its paper's linked GitHub repository returned 404 during this audit; implementation availability remains unresolved. [@stiPose]

A separate monocular CPU method uses perceptual hashing to shortlist binary silhouette templates, then matches using overlap. It demonstrates that this approach is implementable without a GPU, but its reported example is about 0.3 seconds per image and roughly 10-degree rotation error, not evidence of smooth, accurate live tracking for our objects. Its authors identify indistinguishable silhouettes as a limitation. [@silhouetteHash]

For uncolored objects, CNOS is an existing RGB CAD-based localization component: it matches proposed image segments against rendered CAD templates using learned descriptors. It could supply boxes/masks to a pose estimator. It is not itself a six-degree-of-freedom estimator. [@cnos]

This supersedes the initial manual-initialization recommendation below. M3T alone does not satisfy automatic acquisition. A practical first comparison is color segmentation plus silhouette pose search versus color segmentation plus MegaPose. Neither composite has been validated here.

## Shortlist

| Candidate | Inputs and initialization | Assessment |
| --- | --- | --- |
| M3T through Agimus pym3t | RGB; optional depth; initial pose or manual four-point initialization | First classical tracking experiment. MIT upstream; C++ with OpenGL rendering dependencies. CPU optimization does not mean graphics-free operation. [@m3tReadme; @m3tLicense; @pym3t] |
| MegaPose | RGB, calibration, mesh and object bounding box; depth optional | First learned RGB pose-estimation experiment. Novel objects need no object-specific retraining. Apache-2.0 code. Pose estimation is not a complete temporal tracking/recovery system. [@megapose; @megaposeLicense] |
| GigaPose | RGB, rendered mesh templates and object detections | Alternative RGB initializer using template and patch correspondences. MIT repository; benchmark-oriented setup and MegaPose refinement increase integration work. [@gigapose] |
| FoundationPose | RGB-D, calibration, mesh, initial mask | Strong depth-based estimation plus tracking candidate. Distinguish research implementation from newer inference runtime. [@foundationDemo; @foundationRuntime] |
| SAM-6D | RGB-D, mesh and calibration | Includes segmentation and pose estimation, addressing object discovery. Custom demo accepts a PLY CAD path. Temporal tracking and complete distribution licensing were not audited. [@sam6d] |
| ViSP | Prepared CAO/VRML model, calibrated images, initialized pose; optional depth | Mature classical alternative for edge/texture/depth tracking. GPL-2.0-or-later code. Mesh preparation requires investigation. [@visp; @vispLicense] |
| RBOT | Monocular RGB and initialized model | Historical region-based baseline; C++/OpenGL/OpenCV/Qt stack makes it less attractive than M3T for the first prototype. [@rbot] |

These rankings are engineering judgments, not benchmark rankings. Dependency and model licenses need separate review before packaging.

## What “give it an STL” really requires

An STL describes a triangular surface, without standard photographic texture. Conversion to OBJ or PLY can adapt file formats but cannot supply missing appearance information. [@stlFormat] M3T explicitly accepts OBJ geometry with a unit scale and geometry-to-body transform. MegaPose documents PLY/OBJ inputs. [@m3tBody; @megapose]

Proposed minimum input contract:

- Mesh path and explicit mesh units or scale.
- Camera intrinsics and a distortion-handling policy.
- A starting pose for a local tracker, or a bounding box/mask for a pose estimator.
- Optional aligned depth, only for backends that use it.

Agimus's webcam example accepts frames through Python, uses a fixed initial pose, and exposes a 4-by-4 body transform. Its demonstration intrinsics are approximate. This example does not meet the required automatic acquisition workflow. [@pym3tExample]

Geometric implications: wrong mesh scale biases metric translation; rotationally symmetric geometry can leave orientation ambiguous; a hidden object provides no current visual evidence. A quaternion encodes a rotation but does not remove these ambiguities. Test textureless prints, occlusion by hands, motion blur, and similar backgrounds explicitly.

## FoundationPose deployment distinction

The original NVlabs demo registers with RGB, depth, intrinsics and a mask, then tracks subsequent RGB-D frames. Its research source license limits use to noncommercial research/evaluation. [@foundationDemo; @foundationLicense]

The newer NVIDIA inference library exposes Python and C interfaces, uses TensorRT/CUDA, and returns object-to-camera poses. Its documented stack uses CUDA 13.2 containers and driver 580+, with OBJ/PLY input. Its code is Apache-2.0; optional nvdiffrast has separate terms. Model weights identify the NVIDIA Open Model License. Native Windows integration was not verified. Do not transfer the research repository's license conclusion to this separate runtime. [@foundationRuntime; @foundationWeights]

## Fit with the inspected SkellyTracker checkout

Inspected branch: `sfn26`, initially clean. Core currently uses an editable path for SkellyTracker and records `sfn26` under `tool.freemocap.workspace.sources`. No dependency changes are proposed.

Local evidence:

- `skellytracker/core/detectors/detector_base_classes.py`: object detectors return boxes; keypoint detectors return named points.
- `skellytracker/core/data_primitives/observation.py`: stage results hold boxes, keypoints and children, without a general rigid-pose field.
- `skellytracker/core/detectors/keypoint_detectors/charuco/charuco_board_pose.py`: existing calibrated board pose utility returns rotation/translation from known correspondences. It does not match arbitrary CAD geometry.
- `pyproject.toml`: Python 3.11–3.12 and NumPy 2.4.6.

Proposal: introduce an explicit rigid-pose result rather than encoding position/quaternion as keypoints. Include mesh identity, timestamp, translation units, quaternion order, transform direction, and status (initializing/tracking/lost). Keep per-camera/per-object temporal state isolated. Record a backend score only with documented semantics; do not imply it is a calibrated probability.

Define `T_camera_from_object` by `p_camera = R * p_object + t`. Preserve the original mesh origin when converting/centering geometry. Convert rotation matrices to the workspace's wxyz convention at a documented boundary. Camera-relative output is not automatically FreeMoCap world-space output. Core owns cross-repository/world-frame composition.

Proposed configuration:

```text
tracker_type: stl
backend: silhouette_search_candidate
mesh_path: object.stl
mesh_units: mm
camera_calibration: ...
localization: color_mask
initialization: automatic_search
```

Agimus's wrapper needs a native build; its example extra pins NumPy below 2. Compatibility with this checkout's NumPy 2.4.6 and Windows packaging is unverified. Do not install its example environment directly into SkellyTracker. [@pym3tMetadata]

## Recommended evaluation

1. Use one asymmetric rigid object with a known-size STL and a calibrated RGB recording.
2. Automatically extract a mask and box from a distinctive object color; test lighting changes and hand occlusion.
3. Evaluate silhouette template search and refinement from the mesh and measured calibration. Compare against MegaPose using the automatically extracted box.
4. Require automatic initial acquisition and recovery after the object leaves and returns. Consider a temporal tracker only after acquisition works.
5. Measure initialization time, per-frame latency, drift, recovery after full occlusion, reprojection error and pose error against an independent reference. Include symmetric and textureless objects.
6. Choose a backend after these measurements; package optional dependencies so existing detectors remain usable.

A marker rigidly attached to the object is a useful ground-truth or initialization route if the application allows it. It changes the workflow from markerless mesh recognition.

No detector implementation, application test, or downstream integration is included in this audit. Sources are registered in the root bibliography; citation validation checks provenance structure, not algorithm performance.

