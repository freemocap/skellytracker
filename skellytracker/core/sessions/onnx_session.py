"""Generic ONNX Runtime session for GPU-accelerated detectors.

One OnnxSession per Tracker holds all ONNX models needed by that tracker
(YOLOX, RTMPose, or any other ONNX-based detector). Models are identified
by name; detectors look up their model via ``get_session(model_name)``.

Usage::

    session_config = OnnxSessionConfig(
        batch_size=1,
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
import gc
import hashlib
import importlib.util
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from skellytracker.core.sessions.execution_provider_name import ExecutionProviderName
from skellytracker.core.sessions.model_registry import resolve_model_path
from skellytracker.core.sessions.onnx_runtime_protocol import (
    OnnxResourceOwner,
    OnnxRuntimeSession,
)
from skellytracker.core.sessions.ort_session_utils import (
    auto_detect_provider,
    build_tuned_ort_session,
    cuda_device_total_bytes,
    require_provider,
    select_best_cuda_device,
)
from skellytracker.core.sessions.session import Session
from skellytracker.core.sessions.session_errors import SessionCreationError

logger = logging.getLogger(__name__)

_ARENA_VRAM_FRACTION = 0.85


from skellytracker.core.sessions.onnx_model_spec import (
    OnnxModelSpec,
    OnnxSessionConfig,
)


@dataclass
class OnnxSession(Session):
    """Manages all ONNX Runtime sessions for a Tracker's ONNX-based detectors.

    Detectors call ``get_session(model_name)`` to obtain their pre-loaded
    ``ort.InferenceSession``. This avoids redundant CUDA-context creation when
    multiple detectors share the same GPU.
    """

    kind: ClassVar[str] = "onnx"

    _sessions: dict[str, OnnxRuntimeSession] = field(default_factory=dict)
    execution_provider: ExecutionProviderName = "cpu"
    device_id: int = 0
    batch_size: int = 1
    _creation_config: OnnxSessionConfig | None = field(default=None, repr=False)
    _model_paths: dict[str, Path] = field(default_factory=dict, repr=False)
    _model_hashes: dict[str, str] = field(default_factory=dict, repr=False)
    _prepared_warmed: bool = field(default=False, repr=False)
    _resource_owner: OnnxResourceOwner | None = field(default=None, repr=False)

    @property
    def creation_config(self) -> OnnxSessionConfig | None:
        """Return a defensive copy of the configuration used to load these models."""
        return self._creation_config.model_copy(deep=True) if self._creation_config is not None else None

    @classmethod
    def create(cls, config: OnnxSessionConfig, *, engine_cache_dir: Path | None = None,
               progress: Callable[[str, str], None] | None = None) -> OnnxSession:  # type: ignore[override]
        _verify_ort_install()

        if config.execution_provider is None:
            active_provider = auto_detect_provider()
        else:
            require_provider(
                config.execution_provider
            )  # raises SessionCreationError if unavailable
            active_provider = config.execution_provider

        if active_provider in ("cuda", "trt") and sys.platform == "win32":
            _load_nvidia_dlls_on_windows()
        if active_provider in ("cuda", "trt"):
            ort.preload_dlls()

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
        model_paths: dict[str, Path] = {}
        model_hashes: dict[str, str] = {}
        for spec in config.models:
            if progress is not None:
                progress("loading_models", spec.name)
            model_path = resolve_model_path(
                spec.source,
                expected_filename=spec.expected_filename,
                expected_sha256=spec.expected_sha256,
            )
            if spec.prepare is not None:
                model_path = spec.prepare(model_path)
            model_paths[spec.name] = model_path
            with model_path.open("rb") as model_file:
                model_hashes[spec.name] = hashlib.file_digest(model_file, "sha256").hexdigest()

            try:
                ort_session = build_tuned_ort_session(
                    onnx_path=str(model_path),
                    provider=active_provider,
                    fp16=fp16,
                    log_label=spec.name,
                    max_batch_size=config.batch_size,
                    # Missing-person crops reduce batches during live tracking.
                    # Prepare their full range before inference, not on arrival.
                    # [@ortTensorRTProfiles]
                    trt_set_batch_profile=active_provider == "trt",
                    gpu_mem_limit=gpu_mem_limit,
                    device_id=device_id,
                    coreml_options=spec.coreml_options,
                    intra_op_num_threads=config.intra_op_num_threads,
                    cuda_use_tf32=spec.cuda_use_tf32,
                    engine_cache_dir=engine_cache_dir,
                )
            except Exception as exc:
                sessions.clear()
                raise SessionCreationError(
                    f"Failed to load model {spec.name!r} with provider={active_provider!r}: {exc}"
                ) from exc
            sessions[spec.name] = ort_session
            logger.info(
                "OnnxSession: loaded model %r (provider=%r, device=%d)",
                spec.name,
                active_provider,
                device_id,
            )

        onnx_session = cls(
            _sessions=sessions,
            execution_provider=active_provider,
            device_id=device_id,
            batch_size=config.batch_size,
            _creation_config=config.model_copy(deep=True),
            _model_paths=model_paths,
            _model_hashes=model_hashes,
        )
        try:
            _warmup(onnx_session, config.models, active_provider, batch_size=config.batch_size, progress=progress)
        except BaseException:
            onnx_session.close()
            raise
        return onnx_session

    def validate_prepared_backend(self, candidate: OnnxSession) -> None:
        """Check a warmed same-model CUDA→TRT replacement before ownership moves."""
        if candidate._resource_owner is not None:
            candidate._resource_owner.ensure_ready()
        if self.execution_provider != "cuda" or candidate.execution_provider != "trt":
            raise ValueError("Prepared adoption requires a CUDA to TRT transition")
        if not candidate._prepared_warmed:
            raise ValueError("Prepared backend has not completed crop-range warmup")
        if self.device_id != candidate.device_id or self.batch_size != candidate.batch_size:
            raise ValueError("Prepared backend device or capacity mismatch")
        if not self._model_hashes or self._model_hashes != candidate._model_hashes:
            raise ValueError("Prepared backend model bytes mismatch or unknown")
        if self._creation_config is None or candidate._creation_config is None:
            raise ValueError("Prepared backend configuration is unknown")
        expected = self._creation_config.model_copy(deep=True)
        expected.execution_provider = "trt"
        expected.device_id = self.device_id
        if expected != candidate._creation_config or set(self._sessions) != set(candidate._sessions):
            raise ValueError("Prepared backend configuration mismatch")
        for name, current in self._sessions.items():
            prepared = candidate._sessions[name]
            if "TensorrtExecutionProvider" not in prepared.get_providers():
                raise ValueError(f"Prepared model {name!r} did not load TensorRT")
            for accessor in ("get_inputs", "get_outputs"):
                old_signature = [(item.name, item.type, tuple(item.shape)) for item in getattr(current, accessor)()]
                new_signature = [(item.name, item.type, tuple(item.shape)) for item in getattr(prepared, accessor)()]
                if old_signature != new_signature:
                    raise ValueError(f"Prepared model {name!r} has a different tensor contract")

    def adopt_prepared_backend(self, candidate: OnnxSession) -> OnnxSession:
        """Owner-thread, between-request transfer; return old resources without closing.

        The caller schedules returned resources for cleanup away from inference.
        The candidate is consumed; detector references to this wrapper stay valid.
        """
        self.validate_prepared_backend(candidate)
        retired = OnnxSession(_sessions=self._sessions, _resource_owner=self._resource_owner)
        self._sessions, candidate._sessions = candidate._sessions, {}
        self._resource_owner, candidate._resource_owner = candidate._resource_owner, None
        self.execution_provider = candidate.execution_provider
        self._creation_config = candidate._creation_config
        self._model_paths = candidate._model_paths
        return retired

    def preparation_job(self, ticket):
        """Snapshot CUDA identity; prepare and validate TRT only when the job runs."""
        from skellytracker.core.sessions.onnx_prepared_backend import preparation_job
        return preparation_job(self, ticket)

    def isolated_preparation_job(self, ticket):
        """Explicit preparation whose native session stays in an owned process."""
        from skellytracker.core.sessions.isolated_onnx_preparation import (
            isolated_preparation_job,
        )
        return isolated_preparation_job(self, ticket)

    def get_session(self, model_name: str) -> OnnxRuntimeSession:
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

    def run(
        self,
        model_name: str,
        inputs: dict,
        output_names: list[str] | None = None,
    ) -> list:
        """Run inference for *model_name* and return outputs.

        Wraps ``ort.InferenceSession.run()`` with structured error handling so
        callers never need to import ORT exception types directly.

        Raises
        ------
        VRAMExhaustionError
            If the inference call fails due to GPU out-of-memory.
        InferencePipelineError
            If the ONNX Runtime raises any other error during ``session.run()``.
        """
        from skellytracker.core.sessions.session_errors import (
            InferencePipelineError,
            VRAMExhaustionError,
        )

        ort_session = self.get_session(model_name)
        try:
            return ort_session.run(output_names, inputs)
        except MemoryError as exc:
            raise VRAMExhaustionError(
                f"Out of GPU memory running model {model_name!r} "
                f"(provider={self.execution_provider!r}, device={self.device_id})"
            ) from exc
        except Exception as exc:
            msg = str(exc).lower()
            if any(tok in msg for tok in ("out of memory", "cudamalloc", "alloc failed", " oom")):
                raise VRAMExhaustionError(
                    f"Out of GPU memory running model {model_name!r} "
                    f"(provider={self.execution_provider!r}, device={self.device_id}): {exc}"
                ) from exc
            raise InferencePipelineError(
                f"ONNX Runtime error running model {model_name!r} "
                f"(provider={self.execution_provider!r}): {exc}"
            ) from exc

    def run_batched(
        self,
        model_name: str,
        tensors: dict[str, NDArray[np.float32]],
    ) -> dict[str, list]:
        """Run batched inference across N cameras.

        Each value in ``tensors`` is a single-image tensor of shape (3, H, W).
        All images are stacked into a single (N, 3, H, W) batch, fed through
        the model in one ORT call, and the outputs are split back by camera key.
        N may be smaller than the configured batch size when a detection stage
        excludes unusable crops. This is expected and needs no padding or
        session recreation. The model/provider enforces its supported shapes;
        failures propagate through ``run``. Callers skip inference when N is zero.

        Parameters
        ----------
        model_name:
            Name of the model to run (must be loaded in this session).
        tensors:
            Mapping from camera ID to per-image float32 tensor (3, H, W).
            Order is preserved via ``list(tensors.keys())``.

        Returns
        -------
        dict mapping each camera ID to a list of per-image raw output arrays —
        the same format as a single ``session.run()`` call for that image.
        """
        ordered_keys = list(tensors.keys())
        stacked = np.stack([tensors[k] for k in ordered_keys])  # (N, 3, H, W)
        input_name = self.get_session(model_name).get_inputs()[0].name
        raw_outputs = self.run(model_name, {input_name: stacked})
        # raw_outputs is a list of arrays, each (N, ...) — split per camera.
        return {k: [out[i:i+1] for out in raw_outputs] for i, k in enumerate(ordered_keys)}

    def close(self) -> None:
        """Explicitly tear down ORT sessions rather than relying on GC.

        On macOS, CoreMLExecutionProvider sessions hold Metal/Neural-Engine/GCD
        resources that aren't guaranteed to unwind synchronously when garbage
        collected — GC could even run at interpreter-shutdown time, the worst
        possible moment for native cleanup. Dropping references and forcing a
        collection here makes native teardown happen deterministically during
        the pipeline's own shutdown sequence instead.
        """
        self._sessions.clear()
        if self._resource_owner is not None:
            self._resource_owner.close()
            self._resource_owner = None
        gc.collect()


def _warmup(session: OnnxSession, specs: list[OnnxModelSpec], provider: str, *, batch_size: int = 1,
            progress: Callable[[str, str], None] | None = None) -> None:
    """Run a dummy inference through each model to trigger JIT compilation.

    CoreML (macOS) and TRT compile kernels lazily on the first session.run()
    call. Without warmup this happens on the first real frame, causing a silent
    compilation stall in the middle of the demo loop. Warming up here lets
    compilation happen during session creation with a clear log message.

    The dummy batch is exactly the batch size the session will be run with, so
    a graph that cannot handle that batch size fails here — at session creation
    — rather than several seconds later inside the inference loop.
    """
    import time
    logger.info(
        "OnnxSession: warming up %d model(s) on provider=%r at batch_size=%d ...",
        len(specs), provider, batch_size,
    )
    if provider == "trt":
        logger.info("TensorRT warmup can compile uncached engines and may take several minutes")
    for spec in specs:
        if progress is not None:
            progress("warming_models", spec.name)
        ort_session = session.get_session(spec.name)
        input_h, input_w = spec.input_size  # spec.input_size is always (H, W)
        dummy = np.zeros((batch_size, 3, input_h, input_w), dtype=np.float32)
        input_name = ort_session.get_inputs()[0].name
        t0 = time.perf_counter()
        try:
            ort_session.run(None, {input_name: dummy})
        except Exception as exc:
            raise SessionCreationError(
                f"Warmup inference failed for model {spec.name!r} "
                f"(provider={provider!r}, batch_size={batch_size}, "
                f"input shape={dummy.shape}): {exc}"
            ) from exc
        elapsed = time.perf_counter() - t0
        logger.info("OnnxSession: warmup OK for %r (%.1f s)", spec.name, elapsed)


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
            "Install skellytracker with the `all-cuda` or `all-trt` extra: "
            "`uv sync --extra all-cuda`."
        )
    nvidia_root = Path(spec.submodule_search_locations[0])
    bin_dirs = sorted(nvidia_root.glob("*/bin"))
    if not bin_dirs:
        raise RuntimeError(
            f"Found `nvidia` package at {nvidia_root} but no `nvidia/*/bin` "
            f"subdirectories inside it."
        )

    # The pip TensorRT wheels put their DLLs in a top-level `tensorrt_libs/`
    # package, not under `nvidia/*/bin`, so the glob above misses them. Without
    # this, onnxruntime_providers_tensorrt.dll fails to load on nvinfer_10.dll
    # and ORT quietly runs the session on CUDA instead.
    trt_spec = importlib.util.find_spec("tensorrt_libs")
    if trt_spec is not None and trt_spec.submodule_search_locations:
        bin_dirs.append(Path(trt_spec.submodule_search_locations[0]))

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
