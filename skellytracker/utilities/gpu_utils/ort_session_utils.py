"""
Shared ONNX Runtime session utilities for GPU-accelerated trackers.

Used by both RTMPoseSession and CompositeGPUSession to avoid duplicating
the provider-resolution, session-building, and batched-inference boilerplate.

TRT support is wired in but currently unused — the CUDA path is the focus
for now. When TRT is needed, callers can enable it via provider="trt".
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnx
import onnxruntime as ort
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

ExecutionProviderName = Literal["trt", "cuda", "cpu"]


# =============================================================================
# Provider resolution
# =============================================================================


def resolve_provider(
    *,
    requested: ExecutionProviderName,
    on_missing: Literal["fallback", "raise"] = "fallback",
) -> ExecutionProviderName:
    """Pick the actual EP to use given what's available.

    Falls back trt -> cuda -> cpu unless on_missing="raise".
    """
    available = set(ort.get_available_providers())
    needs = {
        "trt": "TensorrtExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }
    if needs[requested] in available:
        return requested
    if on_missing == "raise":
        raise RuntimeError(
            f"Requested execution_provider={requested!r} but ONNX Runtime "
            f"only sees providers={sorted(available)}. Install onnxruntime-gpu "
            f"(and a TensorRT-enabled build for trt) to enable GPU execution."
        )
    fallback_order: list[ExecutionProviderName] = ["trt", "cuda", "cpu"]
    start = fallback_order.index(requested)
    for candidate in fallback_order[start:]:
        if needs[candidate] in available:
            if candidate != requested:
                logger.warning(
                    f"Requested execution_provider={requested!r} not available "
                    f"({sorted(available)}); falling back to {candidate!r}."
                )
            return candidate
    raise RuntimeError(f"No supported ONNX Runtime providers found: {sorted(available)}")


# =============================================================================
# Provider options
# =============================================================================


def cuda_provider_options(*, gpu_mem_limit: int = 2 * 1024 * 1024 * 1024) -> dict:
    """CUDA EP options with exhaustive algorithm search and sensible defaults.

    gpu_mem_limit defaults to 2 GiB — enough for multiple model sessions
    (body + hand + face) sharing a single GPU.
    """
    return {
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "arena_extend_strategy": "kSameAsRequested",
        "do_copy_in_default_stream": True,
        "gpu_mem_limit": gpu_mem_limit,
    }


# =============================================================================
# TRT engine cache management (wired in, activated later)
# =============================================================================


def _default_engine_cache_dir() -> Path:
    return Path.home() / ".cache" / "skellytracker" / "trt_engines"


def validate_engine_cache(engine_cache_dir: Path) -> None:
    """Delete stale TRT engine files when TRT or ORT version has changed.

    Writes a cache_manifest.json recording the current versions. On the next
    call, if stored versions differ from current, all .engine and .timing files
    are deleted so ORT/TRT will recompile fresh engines.
    """
    import json

    manifest_path = engine_cache_dir / "cache_manifest.json"

    try:
        import tensorrt as _trt
        current_trt = _trt.__version__
    except Exception:
        current_trt = "unavailable"
    current = {"trt_version": current_trt, "ort_version": ort.__version__}

    if manifest_path.exists():
        try:
            stored = json.loads(manifest_path.read_text())
        except Exception:
            stored = {}
        if stored != current:
            logger.warning(
                f"TRT engine cache version mismatch "
                f"(stored={stored}, current={current}). "
                f"Deleting stale engines in {engine_cache_dir} — "
                f"they will be recompiled on this run."
            )
            for stale in engine_cache_dir.glob("*.engine"):
                stale.unlink(missing_ok=True)
            for stale in engine_cache_dir.glob("*.timing"):
                stale.unlink(missing_ok=True)

    manifest_path.write_text(json.dumps(current, indent=2))


def _trt_dynamic_batch_profile(
    *,
    onnx_path: str,
    max_batch_size: int,
) -> dict[str, str]:
    """Build TensorRT optimization profile shape strings.

    TRT requires explicit min/opt/max shapes when any input dim is dynamic.
    Pins H/W from the ONNX and lets batch range from 1 to max_batch_size.
    """
    model = onnx.load(onnx_path)
    inp = model.graph.input[0]
    name = inp.name
    dims = inp.type.tensor_type.shape.dim
    fixed_shape = []
    for i, d in enumerate(dims):
        if i == 0:
            continue
        if d.HasField("dim_value") and d.dim_value > 0:
            fixed_shape.append(int(d.dim_value))
        else:
            logger.warning(
                f"TRT profile: input dim {i} of {name!r} is non-static; "
                f"skipping dynamic-batch profile."
            )
            return {}
    fixed_str = "x".join(str(x) for x in fixed_shape)
    min_str = f"{name}:1x{fixed_str}"
    opt_str = f"{name}:{max_batch_size}x{fixed_str}"
    max_str = f"{name}:{max_batch_size}x{fixed_str}"
    logger.info(
        f"TRT optimization profile for {name!r}: "
        f"min={min_str.split(':')[1]}, opt={opt_str.split(':')[1]}, "
        f"max={max_str.split(':')[1]}"
    )
    return {
        "trt_profile_min_shapes": min_str,
        "trt_profile_opt_shapes": opt_str,
        "trt_profile_max_shapes": max_str,
    }


# =============================================================================
# Session construction
# =============================================================================


def build_tuned_ort_session(
    *,
    onnx_path: str,
    provider: ExecutionProviderName,
    engine_cache_dir: Path | None = None,
    fp16: bool = True,
    log_label: str = "model",
    max_batch_size: int | None = None,
    trt_set_batch_profile: bool = False,
    gpu_mem_limit: int = 2 * 1024 * 1024 * 1024,
) -> ort.InferenceSession:
    """Construct an ORT session with explicit SessionOptions + provider options.

    Args:
        onnx_path: Path to the ONNX model file (caller is responsible for any
                   graph surgery like dynamic-batch rewriting before passing).
        provider: Which execution provider to request.
        engine_cache_dir: TRT engine cache directory (defaults to
                          ~/.cache/skellytracker/trt_engines).
        fp16: Enable FP16 mode for TRT.
        log_label: Human-readable label for log messages.
        max_batch_size: When set with trt_set_batch_profile=True, configures
                        TRT optimization profile for this batch range.
        trt_set_batch_profile: If True, set TRT dynamic-batch optimization
                               profile. Requires max_batch_size to be set.
        gpu_mem_limit: GPU memory limit in bytes for CUDA EP arena.
    """
    if engine_cache_dir is None:
        engine_cache_dir = _default_engine_cache_dir()

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    if trt_set_batch_profile:
        engine_cache_dir = engine_cache_dir / "dynbatch_v1"
        engine_cache_dir.mkdir(parents=True, exist_ok=True)

    if provider == "trt":
        validate_engine_cache(engine_cache_dir)

    providers: list[tuple[str, dict] | str] = []
    if provider == "trt":
        trt_options: dict[str, Any] = {
            "trt_fp16_enable": fp16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(engine_cache_dir),
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": str(engine_cache_dir),
            "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
        }
        if trt_set_batch_profile and max_batch_size is not None:
            trt_options.update(
                _trt_dynamic_batch_profile(
                    onnx_path=onnx_path,
                    max_batch_size=max(1, max_batch_size),
                )
            )
        providers.append(("TensorrtExecutionProvider", trt_options))
        providers.append(("CUDAExecutionProvider", cuda_provider_options(gpu_mem_limit=gpu_mem_limit)))
        providers.append("CPUExecutionProvider")
    elif provider == "cuda":
        providers.append(("CUDAExecutionProvider", cuda_provider_options(gpu_mem_limit=gpu_mem_limit)))
        providers.append("CPUExecutionProvider")
    else:
        providers.append("CPUExecutionProvider")

    provider_names = [p if isinstance(p, str) else p[0] for p in providers]
    logger.info(f"Building tuned ORT session for {log_label!r} with providers={provider_names}")

    if provider == "trt":
        logger.info(
            f"  TRT: building {log_label!r} session "
            f"(engine cache: {engine_cache_dir}) — "
            f"first-run TRT compilation can take 1-5 minutes; "
            f"subsequent runs load from cache instantly."
        )

    t0 = time.perf_counter()
    session = ort.InferenceSession(
        path_or_bytes=onnx_path,
        sess_options=sess_options,
        providers=providers,
    )
    elapsed_s = time.perf_counter() - t0
    actual = session.get_providers()
    actual_string = ", ".join(map(str, actual))
    logger.info(f"  {log_label!r} session ready in {elapsed_s:.1f}s (active providers: {actual_string})")
    if provider == "trt" and elapsed_s > 30:
        logger.info(
            f"  TRT engine for {log_label!r} compiled and cached to {engine_cache_dir} — "
            f"next run will load in seconds."
        )
    return session


# =============================================================================
# Batch inference helpers
# =============================================================================


def probe_supports_batch(session: ort.InferenceSession, label: str = "") -> bool:
    """Return True if the session's first input has a dynamic (or > 1) batch dim."""
    try:
        first_input_shape = session.get_inputs()[0].shape
    except Exception:
        return True
    if not first_input_shape:
        return True
    batch_dim = first_input_shape[0]
    if isinstance(batch_dim, str):
        return True
    supports = int(batch_dim) > 1
    if not supports:
        logger.debug(
            f"{label!r} ONNX model has static batch_size={batch_dim}; "
            f"per-image inference will be used."
        )
    return supports


def session_run_batched(
    session: ort.InferenceSession, batch: NDArray
) -> list[NDArray]:
    """Run an ORT session with a batched input. Wraps the standard session.run
    boilerplate for multi-image inference."""
    sess_input_name = session.get_inputs()[0].name
    sess_output_names = [o.name for o in session.get_outputs()]
    return session.run(sess_output_names, {sess_input_name: batch})


# =============================================================================
# Windows GPU DLL helpers
# =============================================================================


def ensure_cuda_dlls_loaded() -> None:
    """Ensure NVIDIA CUDA/cuDNN DLLs are discoverable on Windows.

    Delegates to the existing helper in rtmpose_detector to avoid duplication.
    """
    if sys.platform != "win32":
        ort.preload_dlls()
        return
    from skellytracker.trackers.rtmpose_tracker.rtmpose_detector import (
        _make_nvidia_pip_dlls_discoverable_on_windows,
    )

    _make_nvidia_pip_dlls_discoverable_on_windows()
    ort.preload_dlls()
