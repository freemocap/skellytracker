"""Lightweight ONNX model + session config descriptors.

Split out from onnx_session.py so that detectors (RTMPose, YOLOX) can import
OnnxModelSpec / OnnxSessionConfig without dragging in onnxruntime (a
multi-second import) at sub-process startup. OnnxSession - the class that
actually holds an ORT session - stays in onnx_session.py, which is imported
lazily when a session is created.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from beartype.typing import Callable
from pydantic import ConfigDict, field_validator

from skellytracker.core.config.session_config import SessionConfig
from skellytracker.core.sessions.execution_provider_name import ExecutionProviderName
from skellytracker.core.sessions.model_registry import ModelSource

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
        (e.g. YOLOX dynamic-batch surgery). Applied identically for every
        execution provider. ``None`` = load as-is.
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
    coreml_options: dict | None = None


class OnnxSessionConfig(SessionConfig):
    """Config for OnnxSession.

    Parameters
    ----------
    batch_size:
        Number of frames to process per inference call. **Required** — the
        session fails to construct if omitted. Single-camera setups use 1;
        multi-camera setups pass the camera count. Should equal the number of
        cameras passed to ``Tracker.process_batch()`` / ``DetectionStage.run_batch()``;
        a mismatch triggers a runtime warning from ``run_batched``.
    models:
        List of models to load. Detectors reference them by name.
    execution_provider:
        Which ONNX Runtime execution provider to use. ``None`` = auto-select
        (best available: trt → cuda → coreml on macOS → cpu). When explicitly
        set, ``OnnxSession.create()`` raises ``SessionCreationError`` immediately
        if that provider is unavailable — there is no silent fallback.
    device_id:
        Which GPU to use. ``None`` = auto-select the device with the most free VRAM.
    fp16:
        Enable FP16 mode for TRT EP.
    gpu_mem_limit:
        CUDA arena ceiling in bytes. ``None`` = auto-size from the selected
        device's total VRAM.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: Literal["onnx"] = "onnx"
    batch_size: int
    models: list[OnnxModelSpec] = []
    execution_provider: ExecutionProviderName | None = None
    device_id: int | None = None
    fp16: bool = True
    gpu_mem_limit: int | None = None

    @field_validator("batch_size")
    @classmethod
    def _batch_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"batch_size must be >= 1, got {v}")
        return v

