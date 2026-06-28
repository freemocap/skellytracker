"""
Microbench for RTMPoseSession — validates that batched inference within a
single CUDA context is materially faster than per-image inference.

Run:
    uv run python -m skellytracker.trackers.rtmpose_tracker.bench_rtmpose_session
        --mode lightweight --provider cuda --batch-sizes 1 2 3 4 8

What this measures (intentionally NOT the full pipeline):
  - Single-image latency (predict_single in a loop).
  - Batched latency at various batch sizes (predict_batch).
  - Per-image-equivalent latency for each batch (`batch_ms / N`).

Pipeline-level validation (3 camera processes vs 1 inference node) requires
running the full freemocap app and reading the `Pipeline Timing Report` lines
in the logs — see the `bench_realtime_inference` smoke test in freemocap.

Why this matters: the legacy realtime pipeline runs N camera processes that
each construct their own ONNX session, producing N CUDA contexts on one GPU.
That serializes work and ~triples per-camera latency vs the single-process
case. This bench shows what one tuned session can do when properly batched.
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Literal

import numpy as np

from skellytracker.trackers.old.rtmpose_tracker.rtmpose_session import (
    RTMPoseSession,
    RTMPoseSessionConfig,
)

logger = logging.getLogger(__name__)


def _summary(label: str, samples_ms: list[float]) -> str:
    arr = np.asarray(samples_ms)
    return (
        f"{label:32s}  n={len(arr):4d}  "
        f"mean={arr.mean():7.2f}ms  "
        f"p50={np.percentile(arr, 50):7.2f}ms  "
        f"p95={np.percentile(arr, 95):7.2f}ms  "
        f"max={arr.max():7.2f}ms"
    )


def _make_synthetic_image(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def run(
        *,
        mode: Literal["performance", "lightweight", "balanced"],
        provider: Literal["trt", "cuda", "cpu"],
        batch_sizes: list[int],
        iterations: int,
        image_h: int,
        image_w: int,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")

    print(
        f"\nRTMPoseSession bench — mode={mode!r}, provider={provider!r}, "
        f"image_shape=({image_h}, {image_w}), iters_per_size={iterations}\n"
    )
    session = RTMPoseSession.create(
        RTMPoseSessionConfig(
            mode=mode,
            execution_provider=provider,
            max_batch_size=max(batch_sizes),
            warmup_image_shape=(image_h, image_w),
        ),
    )
    print(f"active provider: {session.active_provider!r}\n")

    # Pre-generate synthetic images so we don't measure RNG cost.
    pool = [_make_synthetic_image(image_h, image_w, seed=i) for i in range(max(batch_sizes) * 2)]

    # ---- Single-image baseline (no batching) ----
    single_samples_ms: list[float] = []
    for i in range(iterations):
        t0 = time.perf_counter()
        session.predict_single(pool[i % len(pool)])
        single_samples_ms.append((time.perf_counter() - t0) * 1e3)
    print(_summary("predict_single (no batch)", single_samples_ms))

    # ---- Batched ----
    for n in batch_sizes:
        if n < 1:
            continue
        batch_samples_ms: list[float] = []
        for i in range(iterations):
            batch = [pool[(i + k) % len(pool)] for k in range(n)]
            t0 = time.perf_counter()
            session.predict_batch(batch)
            batch_samples_ms.append((time.perf_counter() - t0) * 1e3)

        per_image = np.asarray(batch_samples_ms) / n
        arr = np.asarray(batch_samples_ms)
        print(
            f"predict_batch(N={n:<2d})            "
            f"  n={len(arr):4d}  "
            f"mean={arr.mean():7.2f}ms  "
            f"p50={np.percentile(arr, 50):7.2f}ms  "
            f"p95={np.percentile(arr, 95):7.2f}ms  "
            f"per_image_mean={per_image.mean():6.2f}ms"
        )

    print(
        f"\nReference: legacy 3-camera pipeline measured ~71 ms/frame per camera "
        f"on rtmw-dw-l-m. A single-context batched session should beat that on "
        f"the per-image-mean column above.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["performance", "lightweight", "balanced"], default="lightweight")
    parser.add_argument("--provider", choices=["trt", "cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--image-h", type=int, default=720)
    parser.add_argument("--image-w", type=int, default=1280)
    args = parser.parse_args()
    run(
        mode=args.mode,
        provider=args.provider,
        batch_sizes=args.batch_sizes,
        iterations=args.iterations,
        image_h=args.image_h,
        image_w=args.image_w,
    )


if __name__ == "__main__":
    main()
