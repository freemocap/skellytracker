"""Generic ONNX Runtime session for GPU-accelerated detectors.

One OnnxSession per Tracker holds all ONNX models needed by that tracker
(YOLOX, RTMPose, or any other ONNX-based detector). Models are identified
by name; detectors look up their model via ``get_session(model_name)``.

Usage::

    session_config = OnnxSessionConfig(
        models=[
            OnnxModelSpec(name="yolox_m", source=ModelSource(url=MODEL_URLS["yolox-m"]),
                          input_size=(640, 640), prepare=ensure_dynamic_batch),
            OnnxModelSpec(name="rtmw-x-l_256x192",
                          source=ModelSource(url=MODEL_URLS["rtmw-x-l_256x192"]),
                          input_size=(256, 192)),
        ],
    )
    session = OnnxSession.create(session_config)
"""
from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import onnxruntime as ort
from pydantic import ConfigDict

from skellytracker.core.config.session_config import SessionConfig
from skellytracker.core.sessions.session import Session
from skellytracker.core.sessions.execution_provider_name import ExecutionProviderName
from skellytracker.core.sessions.model_registry import ModelSource, resolve_model_path
from skellytracker.core.sessions.ort_session_utils import (
    build_tuned_ort_session,
    cuda_device_total_bytes,
    resolve_provider,
    select_best_cuda_device,
)

logger = logging.getLogger(__name__)

_ARENA_VRAM_FRACTION = 0.85


def _default_provider() -> ExecutionProviderName:
    return "coreml" if sys.platform == "darwin" else "cuda"


@dataclass
class OnnxModelSpec:
    """Descriptor for one ONNX model to load into an OnnxSession.

    Attributes
    ----------
    name:
        Key used by detectors to look up this model via ``session.get_session(name)``.
    source:
        Where to obtain the model file (URL, HF Hub, or local path).
    input_size:
        Model spatial input size ``(H, W)``, used for warmup and provider hints.
    prepare:
        Optional callable applied to the downloaded model path before loading.
        Receives the raw path and returns the path to use for the ORT session
        (e.g. YOLOX dynamic-batch surgery). ``None`` = load as-is.
    coreml_prepare:
        Optional callable used instead of ``prepare`` when the active provider
        is CoreML. Useful when a model needs different graph surgery for CoreML
        (e.g. YOLOX strips the NMS subgraph so CoreML can compile the backbone).
        ``None`` = skip prepare entirely for CoreML.
    coreml_options:
        Provider options dict passed to ``CoreMLExecutionProvider`` when this
        model is loaded. ``None`` = use default CoreML options (no
        ``MLComputeUnits`` override, letting ORT/CoreML pick the best path).
        Set to ``{"MLComputeUnits": "CPUAndGPU"}`` for models whose Neural
        Engine compilation fails with error -5.
    """

    name: str
    source: ModelSource
    input_size: tuple[int, int]
    prepare: Callable[[Path], Path] | None = None
    coreml_prepare: Callable[[Path], Path] | None = None
    coreml_options: dict | None = None


class OnnxSessionConfig(SessionConfig):
    """Config for OnnxSession.

    Parameters
    ----------
    models:
        List of models to load. Detectors reference them by name.
    execution_provider:
        Which ONNX Runtime execution provider to use. ``None`` = auto-select
        (CoreML on macOS, CUDA elsewhere).
    device_id:
        Which GPU to use. ``None`` = auto-select the device with the most free VRAM.
    fp16:
        Enable FP16 mode for TRT EP.
    on_provider_missing:
        ``"fallback"`` to silently downgrade when the requested EP is unavailable;
        ``"raise"`` to hard-error instead.
    gpu_mem_limit:
        CUDA arena ceiling in bytes. ``None`` = auto-size from the selected
        device's total VRAM.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: Literal["onnx"] = "onnx"
    models: list[OnnxModelSpec] = []
    execution_provider: ExecutionProviderName | None = None
    device_id: int | None = None
    fp16: bool = True
    on_provider_missing: Literal["fallback", "raise"] = "fallback"
    gpu_mem_limit: int | None = None


@dataclass
class OnnxSession(Session):
    """Manages all ONNX Runtime sessions for a Tracker's ONNX-based detectors.

    Detectors call ``get_session(model_name)`` to obtain their pre-loaded
    ``ort.InferenceSession``. This avoids redundant CUDA-context creation when
    multiple detectors share the same GPU.
    """

    _sessions: dict[str, ort.InferenceSession] = field(default_factory=dict)
    execution_provider: ExecutionProviderName = "cpu"
    device_id: int = 0

    @classmethod
    def create(cls, config: OnnxSessionConfig | None = None) -> OnnxSession:  # type: ignore[override]
        if config is None:
            config = OnnxSessionConfig()

        _verify_ort_install()

        requested = config.execution_provider or _default_provider()

        if requested in ("cuda", "trt") and sys.platform == "win32":
            _load_nvidia_dlls_on_windows()
        if requested in ("cuda", "trt"):
            ort.preload_dlls()

        active_provider = resolve_provider(
            requested=requested,
            on_missing=config.on_provider_missing,
        )

        # CoreML does not support fp16 inputs.
        fp16 = config.fp16 and active_provider != "coreml"

        device_id = config.device_id
        total_vram_bytes: int | None = None
        if device_id is None and active_provider in ("cuda", "trt"):
            device_id, _free, total_vram_bytes = select_best_cuda_device()
        device_id = device_id if device_id is not None else 0

        gpu_mem_limit = config.gpu_mem_limit
        if gpu_mem_limit is None and active_provider in ("cuda", "trt"):
            if total_vram_bytes is None:
                total_vram_bytes = cuda_device_total_bytes(device_id)
            if total_vram_bytes and total_vram_bytes > 0:
                gpu_mem_limit = int(total_vram_bytes * _ARENA_VRAM_FRACTION)
        if gpu_mem_limit is None:
            gpu_mem_limit = 2 * 1024 * 1024 * 1024

        sessions: dict[str, ort.InferenceSession] = {}
        for spec in config.models:
            model_path = resolve_model_path(spec.source)
            if active_provider == "coreml":
                if spec.coreml_prepare is not None:
                    model_path = spec.coreml_prepare(model_path)
                # else: load original model as-is; CoreML batch=1 is fine
            elif spec.prepare is not None:
                model_path = spec.prepare(model_path)

            try:
                ort_session = build_tuned_ort_session(
                    onnx_path=str(model_path),
                    provider=active_provider,
                    fp16=fp16,
                    log_label=spec.name,
                    gpu_mem_limit=gpu_mem_limit,
                    device_id=device_id,
                    coreml_options=spec.coreml_options,
                )
            except Exception as exc:
                if active_provider != "coreml":
                    raise
                logger.warning(
                    "CoreML failed to load model %r (%s); falling back to CPU.",
                    spec.name, exc,
                )
                ort_session = build_tuned_ort_session(
                    onnx_path=str(model_path),
                    provider="cpu",
                    fp16=False,
                    log_label=spec.name,
                    gpu_mem_limit=gpu_mem_limit,
                    device_id=device_id,
                )
            sessions[spec.name] = ort_session
            logger.info("OnnxSession: loaded model %r (provider=%r, device=%d)", spec.name, active_provider, device_id)

        onnx_session = cls(
            _sessions=sessions,
            execution_provider=active_provider,
            device_id=device_id,
        )
        _warmup(onnx_session, config.models, active_provider)
        return onnx_session

    def get_session(self, model_name: str) -> ort.InferenceSession:
        """Return the ORT session for *model_name*.

        Raises
        ------
        KeyError
            If no model with that name was loaded. Check that the model is
            listed in ``OnnxSessionConfig.models``.
        """
        try:
            return self._sessions[model_name]
        except KeyError as exc:
            available = list(self._sessions.keys())
            raise KeyError(
                f"No ONNX model named {model_name!r} is loaded in this session. "
                f"Available models: {available}. "
                f"Add an OnnxModelSpec with name={model_name!r} to OnnxSessionConfig.models."
            ) from exc

    def close(self) -> None:
        self._sessions.clear()


def _warmup(session: OnnxSession, specs: list[OnnxModelSpec], provider: str) -> None:
    """Run a dummy inference through each model to trigger JIT compilation.

    CoreML (macOS) and TRT compile kernels lazily on the first session.run()
    call. Without warmup this happens on the first real frame, causing a silent
    hang of 5–30 seconds in the middle of the demo loop. Warming up here lets
    compilation happen during session creation with a clear log message.
    """
    import time
    logger.info(
        "OnnxSession: warming up %d model(s) on provider=%r "
        "(first run may take 5–30 s on CoreML/TRT) ...",
        len(specs), provider,
    )
    for spec in specs:
        ort_session = session.get_session(spec.name)
        input_h, input_w = spec.input_size  # spec.input_size is always (H, W)
        dummy = np.zeros((1, 3, input_h, input_w), dtype=np.float32)
        input_name = ort_session.get_inputs()[0].name
        t0 = time.perf_counter()
        try:
            ort_session.run(None, {input_name: dummy})
            elapsed = time.perf_counter() - t0
            logger.info("OnnxSession: warmup OK for %r (%.1f s)", spec.name, elapsed)
        except Exception as exc:
            logger.warning("OnnxSession: warmup failed for %r: %r (non-fatal)", spec.name, exc)


def _verify_ort_install() -> None:
    try:
        providers = ort.get_available_providers()
    except AttributeError as e:
        raise RuntimeError(
            "ONNX Runtime install appears broken — "
            "`onnxruntime.get_available_providers()` is missing. "
            "This usually means CPU `onnxruntime` and GPU `onnxruntime-gpu` "
            "collided on disk. Reinstall a single build, e.g. "
            "`pip install --force-reinstall onnxruntime-gpu`."
        ) from e
    if not providers:
        raise RuntimeError("ONNX Runtime reports no execution providers. The install is broken.")


def _load_nvidia_dlls_on_windows() -> None:
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "Could not find `nvidia` pip namespace package. "
            "Install skellytracker with the `rtmpose-gpu` extra: "
            "`uv sync --extra rtmpose-gpu`."
        )
    nvidia_root = Path(spec.submodule_search_locations[0])
    bin_dirs = sorted(nvidia_root.glob("*/bin"))
    if not bin_dirs:
        raise RuntimeError(
            f"Found `nvidia` package at {nvidia_root} but no `nvidia/*/bin` "
            f"subdirectories inside it."
        )

    bin_dir_strs = [str(d) for d in bin_dirs]
    os.environ["PATH"] = os.pathsep.join([*bin_dir_strs, os.environ.get("PATH", "")])

    for bin_dir in bin_dirs:
        os.add_dll_directory(str(bin_dir))

    cudnn_bin = nvidia_root / "cudnn" / "bin"
    if cudnn_bin.is_dir():
        for dll_path in sorted(cudnn_bin.glob("*.dll")):
            try:
                ctypes.WinDLL(str(dll_path))
            except OSError:
                logger.debug("Could not preload cuDNN DLL: %s", dll_path)
