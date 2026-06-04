"""
Microbench for RtPoseDetector — measures detection and pose estimation step
latency separately so you can track the impact of swapping the detector
(e.g. RT-DETR → YOLO) or moving to a different device (CPU → MPS/CUDA).

Run:
    python -m skellytracker.trackers.rt_pose_tracker.bench_rt_pose --image-path /path/to/photo.jpg
    python -m skellytracker.trackers.rt_pose_tracker.bench_rt_pose --image-path photo.jpg --device cpu --iterations 30
    python -m skellytracker.trackers.rt_pose_tracker.bench_rt_pose --image-path photo.jpg --dtype float32

--image-path should point to a photo that contains at least one person so that
the pose estimation step actually runs. Any common image format works (JPEG, PNG, etc).

What this measures:
  - detection_step:  RT-DETR forward pass + box postprocessing (person filter).
  - pose_step (N=k): VitPose forward pass + heatmap decode for k detected persons.
  - full detect():   end-to-end RtPoseDetector.detect() including BGR→RGB + tensor ops.
  - (warmup frames are discarded; reported samples are stable repeated-inference cost)

Interpreting results:
  - On CPU: expect detection ~200-500ms, pose ~100-300ms per person.
  - On CUDA (bfloat16): expect detection ~10-30ms, pose ~5-20ms.
  - Target for real-time (30 FPS): < 33ms total.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from skellytracker.trackers.rt_pose_tracker.rt_pose_detector import RtPoseDetector, RtPoseDetectorConfig

logger = logging.getLogger(__name__)


def _summary(label: str, samples_ms: list[float]) -> str:
    arr = np.asarray(samples_ms)
    return (
        f"{label:36s}  n={len(arr):4d}  "
        f"mean={arr.mean():8.2f}ms  "
        f"p50={np.percentile(arr, 50):8.2f}ms  "
        f"p95={np.percentile(arr, 95):8.2f}ms  "
        f"max={arr.max():8.2f}ms"
    )


def run(
    *,
    image_path: str,
    device: str | None,
    dtype: str,
    detection_checkpoint: str,
    pose_checkpoint: str,
    iterations: int,
    warmup: int,
) -> None:
    import cv2
    import torch

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(name)s | %(message)s")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path!r}")
    image_h, image_w = image.shape[:2]

    config = RtPoseDetectorConfig(
        object_detection_checkpoint=detection_checkpoint,
        pose_estimation_checkpoint=pose_checkpoint,
        device=device,
        dtype=dtype,
        compile_models=False,
    )

    print(f"\nRtPose bench — device={device or 'auto'!r}, dtype={dtype!r}, "
          f"image=({image_h}×{image_w}), warmup={warmup}, iters={iterations}")
    print(f"  detection:  {detection_checkpoint}")
    print(f"  pose:       {pose_checkpoint}")
    print(f"  image:      {image_path}\n")

    print("Loading models...", flush=True)
    detector = RtPoseDetector.create(config)
    actual_device = detector._device
    detection_device = detector._detection_device
    if detection_device != actual_device:
        print(f"Running on device: detection={detection_device!r}, pose={actual_device!r} (mixed)\n")
    else:
        print(f"Running on device: {actual_device!r}\n")

    # Same frame repeated — identical input isolates model latency from
    # per-frame variability (person count, pose complexity).
    pool = [image] * (max(iterations, warmup) + 4)

    def _to_detection_tensor(image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb.astype(np.float32)).to(detection_device)

    def _to_pose_tensor(image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb.astype(np.float32)).to(actual_device)

    # ── Warmup ────────────────────────────────────────────────────────────────
    if warmup > 0:
        print(f"Warming up ({warmup} frames)...", flush=True)
        for i in range(warmup):
            dt = _to_detection_tensor(pool[i])
            boxes = detector._run_detection(dt)
            if boxes.shape[0] > 0:
                pt = _to_pose_tensor(pool[i])
                detector._run_pose_estimation(pt, boxes.to(actual_device))
        print("Warmup done.\n")

    # ── Detection step ────────────────────────────────────────────────────────
    detection_ms: list[float] = []
    detected_boxes: list[torch.Tensor] = []
    for i in range(iterations):
        dt = _to_detection_tensor(pool[i])
        t0 = time.perf_counter()
        boxes = detector._run_detection(dt)
        detection_ms.append((time.perf_counter() - t0) * 1e3)
        detected_boxes.append(boxes)

    print(_summary("detection_step (RT-DETR)", detection_ms))

    # ── Pose step (using boxes from detection) ────────────────────────────────
    pose_by_n: dict[int, list[float]] = {}
    for i, boxes in enumerate(detected_boxes):
        n = int(boxes.shape[0])
        if n == 0:
            continue
        pt = _to_pose_tensor(pool[i])
        t0 = time.perf_counter()
        detector._run_pose_estimation(pt, boxes.to(actual_device))
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        pose_by_n.setdefault(n, []).append(elapsed_ms)

    if not pose_by_n:
        print("pose_step                             — no persons detected in any frame")
    else:
        for n in sorted(pose_by_n):
            label = f"pose_step (N={n} person{'s' if n != 1 else ''} detected)"
            print(_summary(label, pose_by_n[n]))

    # ── Full end-to-end detect() ──────────────────────────────────────────────
    full_ms: list[float] = []
    for i in range(iterations):
        t0 = time.perf_counter()
        detector.detect(frame_number=i, image=pool[i])
        full_ms.append((time.perf_counter() - t0) * 1e3)

    print(_summary("full detect() end-to-end", full_ms))

    # ── Summary ───────────────────────────────────────────────────────────────
    det_mean = np.mean(detection_ms)
    pose_means = [np.mean(v) for v in pose_by_n.values()]
    pose_mean = np.mean(pose_means) if pose_means else float("nan")
    total_mean = np.mean(full_ms)
    fps_estimate = 1000.0 / total_mean if total_mean > 0 else float("nan")

    print(f"\n{'─'*80}")
    print(f"  device:          {actual_device}")
    print(f"  detection mean:  {det_mean:.1f}ms")
    print(f"  pose mean:       {pose_mean:.1f}ms  (averaged across person counts)")
    print(f"  full mean:       {total_mean:.1f}ms  → ~{fps_estimate:.1f} FPS theoretical max")
    print("  real-time bar:   33ms (30 FPS) / 16ms (60 FPS)")
    print(f"{'─'*80}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-path", required=True, help="Path to an image containing at least one person")
    parser.add_argument("--device", default=None, help="'cpu', 'cuda', 'mps', or omit for auto")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument(
        "--detection-checkpoint",
        default="PekingU/rtdetr_r50vd_coco_o365",
        help="HuggingFace checkpoint for the person detector",
    )
    parser.add_argument(
        "--pose-checkpoint",
        default="usyd-community/vitpose-plus-small",
        help="HuggingFace checkpoint for the pose estimator",
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    run(
        image_path=args.image_path,
        device=args.device,
        dtype=args.dtype,
        detection_checkpoint=args.detection_checkpoint,
        pose_checkpoint=args.pose_checkpoint,
        iterations=args.iterations,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
