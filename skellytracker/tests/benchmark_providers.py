"""Compare CPU/CUDA on identical decoded reference frames.

Run from the repository root with ``python -m skellytracker.tests.benchmark_providers``.
Reports end-to-end tracker latency, not video decoding, annotation, or pure kernel
time. CPU/CUDA differences are numerical agreement, not ground-truth accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from skellytracker.tests.onnx_provider_checks import verify_session_provider


def load_frames(path: Path, count: int):
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        for _ in range(count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"{path}: expected {count} frames, got {len(frames)}")
            frames.append(frame)
    finally:
        capture.release()
    return frames


def frame_digest(frames):
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.tobytes())
    return digest.hexdigest()


def latency_summary(samples):
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0 or np.any(samples <= 0) or not np.all(np.isfinite(samples)):
        raise ValueError("Latency samples must be finite and positive")
    return {
        "frames": int(samples.size),
        "mean_ms": float(samples.mean() * 1000),
        "median_ms": float(np.median(samples) * 1000),
        "p95_ms": float(np.percentile(samples, 95) * 1000),
        "frames_per_second": float(samples.size / samples.sum()),
    }


def output_agreement(cpu, cuda):
    finite_cpu = np.isfinite(cpu[..., :2]).all(axis=-1)
    finite_cuda = np.isfinite(cuda[..., :2]).all(axis=-1)
    common = finite_cpu & finite_cuda
    distances = np.linalg.norm(cpu[..., :2][common] - cuda[..., :2][common], axis=-1)
    return {
        "comparable_keypoints": int(common.sum()),
        "finite_mask_disagreements": int((finite_cpu != finite_cuda).sum()),
        "median_xy_difference_pixels": float(np.median(distances)) if distances.size else None,
        "p95_xy_difference_pixels": float(np.percentile(distances, 95)) if distances.size else None,
        "max_xy_difference_pixels": float(distances.max()) if distances.size else None,
    }


def run_provider(provider, videos, repeats, warmup):
    # Import only when running, so --help and report-helper tests need no runtime.
    from skellytracker.core import DetectionStageConfig, Tracker, TrackerConfig, TrackerState
    from skellytracker.core.detectors.keypoint_detectors.rtmpose import (
        RTMPoseDetectorConfig, RTMPoseKeypointDetector,
    )
    from skellytracker.core.detectors.object_detectors.yolox import (
        YoloxPersonDetector, YoloxPersonDetectorConfig,
    )
    from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig

    specs = [YoloxPersonDetector.model_spec("yolox-m"),
             RTMPoseKeypointDetector.model_spec("rtmw-x-l_256x192")]
    start = time.perf_counter()
    session = OnnxSession.create(OnnxSessionConfig(
        execution_provider=provider, batch_size=1, fp16=False, models=specs,
    ))
    initialization_seconds = time.perf_counter() - start
    tracker = None
    try:
        active = verify_session_provider(session, [spec.name for spec in specs], provider)
        threading = {}
        for spec in specs:
            options = session.get_session(spec.name).get_session_options()
            try:
                spinning = options.get_session_config_entry("session.intra_op.allow_spinning")
            except RuntimeError:
                spinning = "default"
            threading[spec.name] = {
                "intra_op_num_threads": options.intra_op_num_threads,
                "inter_op_num_threads": options.inter_op_num_threads,
                "intra_op_allow_spinning": spinning,
            }
        tracker = Tracker.create(TrackerConfig(stages=[DetectionStageConfig(
            name="body", object_detector=YoloxPersonDetectorConfig(),
            keypoint_detectors=[RTMPoseDetectorConfig()],
        )]), {"onnx": session})
        state = TrackerState()
        first_frames = next(iter(videos.values()))
        for index in range(warmup):
            _, state = tracker.process_image(first_frames[index % len(first_frames)], frame_number=index, state=state)

        samples, coordinates, camera_reports = [], [], {}
        for camera, frames in videos.items():
            camera_samples, detected_frames, runs = [], [], []
            for repeat in range(repeats):
                tracker.reset_temporal_state()
                state = TrackerState()
                detected, run_samples = 0, []
                for index, frame in enumerate(frames):
                    start = time.perf_counter()
                    observation, state = tracker.process_image(frame, frame_number=index, state=state)
                    elapsed = time.perf_counter() - start
                    points = observation.stages["body"].keypoints
                    if points.xyz.shape != (133, 3):
                        raise AssertionError(f"Unexpected keypoint shape: {points.xyz.shape}")
                    if not np.all(np.isfinite(points.visibility) & (points.visibility >= 0) & (points.visibility <= 1)):
                        raise AssertionError("Invalid keypoint visibility")
                    detected += int(points.n_valid > 0)
                    run_samples.append(elapsed)
                    if repeat == 0:
                        coordinates.append(points.xyz.copy())
                if not detected:
                    raise AssertionError(f"{provider}, {camera}: no person detected")
                camera_samples.extend(run_samples)
                detected_frames.append(detected)
                runs.append(latency_summary(run_samples))
            samples.extend(camera_samples)
            camera_reports[camera] = {
                **latency_summary(camera_samples), "runs": runs,
                "frames_with_detection_per_run": detected_frames,
            }
        return {
            "provider": provider, "active_model_providers": active,
            "model_threading": threading,
            "initialization_including_runtime_warmup_seconds": initialization_seconds,
            "device_id": session.device_id, "models": [spec.name for spec in specs],
            "latency": latency_summary(samples), "cameras": camera_reports,
        }, np.stack(coordinates)
    finally:
        if tracker is not None:
            tracker.close()
        else:
            session.close()


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_video_dir = os.environ.get("SKELLYTRACKER_TEST_VIDEO_DIR") or (
        Path.home() / "freemocap_data/testing/prepared/freemocap_test_data/current"
        / "recordings/freemocap_test_data/synchronized_videos"
    )
    parser.add_argument("video_dir", type=Path, nargs="?", default=default_video_dir,
                        help="Reference synchronized_videos directory (defaults to local prepared test data)")
    parser.add_argument("--providers", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    parser.add_argument("--frames", type=positive_integer, default=30)
    parser.add_argument("--repeats", type=positive_integer, default=3)
    parser.add_argument("--warmup", type=positive_integer, default=3)
    parser.add_argument("--output", type=Path, default=Path(".test-artifacts/provider-benchmark"))
    args = parser.parse_args()
    if len(set(args.providers)) != len(args.providers):
        parser.error("Providers must be unique")
    import onnxruntime as ort
    required = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}
    for provider in args.providers:
        if required[provider] not in ort.get_available_providers():
            raise RuntimeError(f"Requested {provider} is unavailable; refusing a partial comparison")
    paths = sorted(args.video_dir.resolve().glob("*.mp4"))
    if not paths:
        parser.error("No reference MP4 videos found")
    # All generated reports must stay in ignored local artifacts.
    check = subprocess.run(["git", "check-ignore", "-q", str(args.output / "results.json")], check=False)
    if check.returncode != 0:
        parser.error("Output directory must be ignored by the current repository")
    args.output.mkdir(parents=True, exist_ok=True)
    videos = {path.name: load_frames(path, args.frames) for path in paths}
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "processor": platform.processor(),
        "python": platform.python_version(), "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "video_dir": str(args.video_dir.resolve()),
        "decoded_frame_sha256": {name: frame_digest(frames) for name, frames in videos.items()},
        "frames_per_camera": args.frames, "repeats": args.repeats, "warmup_frames": args.warmup,
        "batch_size": 1, "fp16": False, "timing_scope": "tracker.process_image; excludes video decoding and annotation",
        "provider_validation": "active session providers; per-node placement is not profiled",
        "workload": "YOLOX-m person detection and RTMW-x-l pose on every frame; no crop reuse",
        "results": {}, "completed": False,
    }
    output = args.output / "results.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    coordinates = {}
    for provider in args.providers:
        print(f"Running {provider} on {len(videos)} cameras...", flush=True)
        result, coordinates[provider] = run_provider(provider, videos, args.repeats, args.warmup)
        report["results"][provider] = result
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(args.output / f"{provider}-keypoints.npz", xyz=coordinates[provider])
        print(json.dumps(result["latency"]), flush=True)
    if "cpu" in coordinates and "cuda" in coordinates:
        report["output_agreement"] = output_agreement(coordinates["cpu"], coordinates["cuda"])
        report["cuda_throughput_ratio_to_cpu"] = (
            report["results"]["cuda"]["latency"]["frames_per_second"] /
            report["results"]["cpu"]["latency"]["frames_per_second"])
    report["completed"] = True
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {output.resolve()}")


if __name__ == "__main__":
    main()
