"""
Shared ONNX Runtime session utilities for GPU-accelerated trackers.

Used by both RTMPoseSession and CompositeGPUSession to avoid duplicating
the provider-resolution, session-building, and batched-inference boilerplate.

TRT support is wired in but currently unused — the CUDA path is the focus
for now. When TRT is needed, callers can enable it via provider="trt".
"""

import ctypes
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


def _load_cudart():
    if sys.platform == "win32":
        candidates = ["cudart64_12.dll", "cudart64_11.dll"]
    elif sys.platform.startswith("linux"):
        candidates = ["libcudart.so.12", "libcudart.so.11.0", "libcudart.so"]
    else:
        logger.debug("CUDA runtime: platform %r not supported -- skipping device query", sys.platform)
        return None
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            logger.debug("CUDA runtime loaded: %r", name)
            return lib
        except OSError:
            continue
    logger.debug("CUDA runtime not found (tried %s) -- device_id will default to 0", candidates)
    return None


def _get_device_name(cudart, device_idx: int) -> str:
    """Query the GPU name via cudaGetDeviceProperties.

    cudaDeviceProp has 'char name[256]' as its very first field, so we can
    read the name by passing an oversized buffer — no need to define the full
    struct (which is ~800 bytes and changes across CUDA versions).
    """
    buf = ctypes.create_string_buffer(8192)
    rc = cudart.cudaGetDeviceProperties(buf, ctypes.c_int(device_idx))
    if rc != 0:
        logger.debug("  device %d: cudaGetDeviceProperties returned error %d", device_idx, rc)
        return f"device {device_idx} (name unavailable)"
    name = buf.raw[:256].rstrip(b"\x00").decode("utf-8", errors="replace").strip()
    return name or f"device {device_idx}"


def _print_device_survey(
    rows: list[tuple[int, str, int, int]],  # (idx, name, total_mib, free_mib)
    best_idx: int,
    best_name: str,
    best_mib: int,
    reason: str,
) -> None:
    """Log a formatted device survey table."""
    n = len(rows)

    # Column widths — name column expands to fit the longest name
    id_w     = 6
    name_w   = max(28, max(len(r[1]) for r in rows) + 4)
    total_w  = 15
    free_w   = 15
    status_w = 20

    ws = [id_w, name_w, total_w, free_w, status_w]

    def _rule(l, mi, r):
        return "  " + l + mi.join("═" * w for w in ws) + r

    top   = _rule("╔", "╦", "╗")
    mid   = _rule("╠", "╬", "╣")
    bot   = _rule("╚", "╩", "╝")

    inner_w = sum(ws) + len(ws) - 1  # total inner width including separators
    title   = f"  CUDA DEVICE SURVEY  —  {n} CUDA-capable device(s) found  "
    title_line = f"  ║{title:<{inner_w}}║"

    hdr = (
        f"  ║{'  ID':<{id_w}}║"
        f"{'  Device Name':<{name_w}}║"
        f"{'  Total VRAM':>{total_w}}║"
        f"{'  Free VRAM':>{free_w}}║"
        f"{'  Status':<{status_w}}║"
    )

    lines = ["", top, title_line, mid, hdr, mid]

    for idx, name, total_mib, free_mib in rows:
        total_str  = f"  {total_mib:>8,} MiB  " if total_mib >= 0 else "    unavailable"
        free_str   = f"  {free_mib:>8,} MiB  "  if free_mib  >= 0 else "    unavailable"
        status_str = "  ✓  SELECTED" if idx == best_idx else ""
        lines.append(
            f"  ║  {idx:<{id_w - 2}}║"
            f"  {name:<{name_w - 2}}║"
            f"{total_str:>{total_w}}║"
            f"{free_str:>{free_w}}║"
            f"{status_str:<{status_w}}║"
        )

    lines += [
        bot,
        "",
        f"  Selected : device_id={best_idx}",
        f"  Name     : {best_name}",
        f"  VRAM     : {best_mib:,} MiB total",
        f"  Reason   : {reason}",
        "",
    ]
    logger.info("\n".join(lines))


def select_best_cuda_device_id() -> int:
    """Pick the NVIDIA GPU with the most total VRAM using the CUDA Runtime C API.

    Uses nvidia-cuda-runtime-cu12 (already a required dep for all GPU extras) via
    ctypes — no subprocess, no nvidia-smi, no extra packages needed.

    Returns 0 on any failure (safe default: same as not calling this at all).
    """
    cudart = _load_cudart()
    if cudart is None:
        return 0

    count = ctypes.c_int(0)
    rc = cudart.cudaGetDeviceCount(ctypes.byref(count))
    if rc != 0:
        logger.debug("cudaGetDeviceCount returned error %d -- using device_id=0", rc)
        return 0

    n = count.value
    logger.debug("cudaGetDeviceCount: %d CUDA device(s) visible", n)

    if n == 0:
        logger.info("CUDA device selection: no CUDA-capable devices found -- using device_id=0")
        return 0

    # Query name + VRAM for every device
    rows: list[tuple[int, str, int, int]] = []  # (idx, name, total_mib, free_mib)
    best_idx, best_bytes = 0, -1

    for i in range(n):
        name = _get_device_name(cudart, i)

        rc = cudart.cudaSetDevice(i)
        if rc != 0:
            logger.debug("  device %d (%s): cudaSetDevice error %d -- skipping", i, name, rc)
            rows.append((i, name, -1, -1))
            continue

        free = ctypes.c_size_t(0)
        total = ctypes.c_size_t(0)
        rc = cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        if rc != 0:
            logger.debug("  device %d (%s): cudaMemGetInfo error %d -- skipping", i, name, rc)
            rows.append((i, name, -1, -1))
            continue

        total_mib = total.value // (1024 * 1024)
        free_mib  = free.value  // (1024 * 1024)
        rows.append((i, name, total_mib, free_mib))
        logger.debug("  device %d (%s): %d MiB total, %d MiB free", i, name, total_mib, free_mib)

        if total.value > best_bytes:
            best_bytes = total.value
            best_idx   = i

    best_name = next((r[1] for r in rows if r[0] == best_idx), f"device {best_idx}")
    best_mib  = best_bytes // (1024 * 1024) if best_bytes >= 0 else 0

    if n == 1:
        reason = "only CUDA device available"
    else:
        reason = f"highest total VRAM of {n} CUDA device(s)"

    _print_device_survey(rows, best_idx, best_name, best_mib, reason)
    return best_idx

ExecutionProviderName = Literal["trt", "cuda", "coreml", "cpu"]


# =============================================================================
# Provider resolution
# =============================================================================

_PROVIDER_EP_NAME: dict[str, str] = {
    "trt": "TensorrtExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def resolve_provider(
    *,
    requested: ExecutionProviderName,
    on_missing: Literal["fallback", "raise"] = "fallback",
) -> ExecutionProviderName:
    """Pick the actual EP to use given what's available.

    Falls back trt -> cuda -> coreml -> cpu unless on_missing="raise".
    CoreML is only available on macOS; it is skipped on other platforms.
    """
    import sys
    available = set(ort.get_available_providers())
    if needs := _PROVIDER_EP_NAME.get(requested):
        if needs in available:
            return requested
    if on_missing == "raise":
        raise RuntimeError(
            f"Requested execution_provider={requested!r} but ONNX Runtime "
            f"only sees providers={sorted(available)}. Install onnxruntime-gpu "
            f"(and a TensorRT-enabled build for trt) to enable GPU execution."
        )
    fallback_order: list[ExecutionProviderName] = ["trt", "cuda", "coreml", "cpu"]
    start = fallback_order.index(requested)
    for candidate in fallback_order[start:]:
        ep = _PROVIDER_EP_NAME[candidate]
        if candidate == "coreml" and sys.platform != "darwin":
            continue
        if ep in available:
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


def cuda_provider_options(*, gpu_mem_limit: int = 2 * 1024 * 1024 * 1024, device_id: int = 0) -> dict:
    """CUDA EP options with exhaustive algorithm search and sensible defaults.

    gpu_mem_limit defaults to 2 GiB -- enough for multiple model sessions
    (body + hand + face) sharing a single GPU.
    device_id selects which CUDA GPU to use (0 = first, 1 = second, ...).
    """
    return {
        "device_id": device_id,
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
    device_id: int = 0,
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
            "trt_device_id": device_id,
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
        providers.append(("CUDAExecutionProvider", cuda_provider_options(gpu_mem_limit=gpu_mem_limit, device_id=device_id)))
        providers.append("CPUExecutionProvider")
    elif provider == "cuda":
        providers.append(("CUDAExecutionProvider", cuda_provider_options(gpu_mem_limit=gpu_mem_limit, device_id=device_id)))
        providers.append("CPUExecutionProvider")
    elif provider == "coreml":
        # CoreML EP uses Metal on Apple Silicon. Dynamic batch dims crash CoreML
        # (SIGSEGV), so callers must use batch_size=1 (RTMPoseSession enforces
        # this via supports_batching=False). fp16 is also unsupported by CoreML.
        providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")
    else:
        providers.append("CPUExecutionProvider")

    provider_names = [p if isinstance(p, str) else p[0] for p in providers]
    logger.info(
        "Building ORT session: label=%r  provider=%r  device_id=%d  providers=%s",
        log_label, provider, device_id, provider_names,
    )

    if provider == "trt":
        logger.info(
            "  [%s] TRT session on device_id=%d (engine cache: %s) -- "
            "first-run compilation can take 1-5 minutes; subsequent runs load from cache instantly.",
            log_label, device_id, engine_cache_dir,
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
    logger.info(
        "  [%s] session ready in %.1fs  device_id=%d  active providers: %s",
        log_label, elapsed_s, device_id, actual_string,
    )
    if provider == "trt" and elapsed_s > 30:
        logger.info(
            "  [%s] TRT engine compiled and cached to %s -- next run will load in seconds.",
            log_label, engine_cache_dir,
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
    session: ort.InferenceSession,
    batch: NDArray,
    *,
    input_name: str | None = None,
    output_names: list[str] | None = None,
) -> list[NDArray]:
    """Run an ORT session with a batched input.

    Prefer passing *input_name* and *output_names* (cached at session build
    time) to avoid per-frame graph-metadata traversal overhead.
    """
    if input_name is None:
        input_name = session.get_inputs()[0].name
    if output_names is None:
        output_names = [o.name for o in session.get_outputs()]
    return session.run(output_names, {input_name: batch})


# =============================================================================
# Dynamic-batch ONNX rewriting
# =============================================================================

_DYNBATCH_PARAM = "N"
_DYNBATCH_SUFFIX = ".dynbatch.onnx"


def ensure_dynamic_batch_onnx(
    src_path: str | Path,
    dst_path: str | Path | None = None,
) -> Path:
    """Rewrite an ONNX model to accept dynamic batch sizes.

    Changes ``dim[0]`` from a static ``dim_value=1`` to the symbolic
    ``dim_param="N"`` on the first graph input and all graph outputs.

    **No Reshape surgery is performed.**  Models whose graph contains
    Reshape nodes that hardcode ``batch=1`` will continue to fail.
    For YOLOX-style models see ``_yolox_dynamic_batch.py`` for the full
    Reshape fixup pipeline.

    Parameters
    ----------
    src_path : Path
        Path to the static-batch ONNX model.
    dst_path : Path or None
        Where to write the modified model.  Defaults to
        ``<src_stem>.dynbatch.onnx`` in the same directory.

    Returns
    -------
    Path
        Path to the dynamic-batch model (cached if it already exists).
    """
    import onnx

    src = Path(src_path)
    dst = Path(dst_path) if dst_path else src.with_suffix("").with_suffix(_DYNBATCH_SUFFIX)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        logger.info(f"Using cached dynamic-batch model: {dst}")
        return dst

    logger.info(f"Rewriting static-batch ONNX → dynamic batch: {src}")
    model = onnx.load(str(src))

    # --- Symbolize batch dim on first input ---
    graph = model.graph
    if graph.input:
        _symbolize_batch_dim(graph.input[0])

    # --- Symbolize batch dim on all outputs ---
    for out in graph.output:
        _symbolize_batch_dim(out)

    onnx.save(model, str(dst))
    logger.info(f"Dynamic-batch model written: {dst}")
    return dst


def _symbolize_batch_dim(value_info) -> None:
    """Replace a hard ``dim_value=1`` on the leading axis with ``dim_param='N'``."""
    tensor_type = value_info.type.tensor_type
    shape = tensor_type.shape
    if not shape.dim:
        return
    leading = shape.dim[0]
    if leading.HasField("dim_value") and leading.dim_value == 1:
        leading.ClearField("dim_value")
        leading.dim_param = _DYNBATCH_PARAM
    elif not leading.HasField("dim_param"):
        leading.dim_param = _DYNBATCH_PARAM


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
