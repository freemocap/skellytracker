"""Glue between a validated `SidecarModel` and the existing detector/session
machinery (`ObjectDetector`/`KeypointDetector`/`OnnxSession`).

Free functions only — sidecar-driven detectors (e.g. `YoloxPersonDetector`,
the RTMW wholebody detector) call into these from their `preprocess`/
`postprocess`/`model_spec` methods rather than sidecars getting their own
detector base classes. This keeps the existing `ObjectDetector`/
`KeypointDetector` ABCs and `OBJECT_DETECTOR_REGISTRY`/
`KEYPOINT_DETECTOR_REGISTRY` as the only plug-in point.
"""
from __future__ import annotations

from pathlib import Path

from beartype.typing import Callable

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.sidecar.model import CustomNormalization, InputSpec, Precision, SidecarModel
from skellytracker.core.sessions.model_registry import ModelSource, resolve_model_path
from skellytracker.core.sessions.onnx_session import OnnxModelSpec

_IMAGENET_MEAN: tuple[float, float, float] = (123.675, 116.28, 103.53)
_IMAGENET_STD: tuple[float, float, float] = (58.395, 57.12, 57.375)


def resolve_normalization_mode(input_spec: InputSpec, precision: Precision) -> str | CustomNormalization:
    """Resolve the effective normalization mode for `precision`.

    Resolution order (see specs/sidecar-spec.md, "Normalization modes"):
    `normalization_by_precision[precision]` -> top-level `normalization` -> default `imagenet_bgr`.
    """
    by_precision = input_spec.normalization_by_precision or {}
    if precision in by_precision:
        return by_precision[precision]
    return input_spec.normalization


def build_normalization_fn(input_spec: InputSpec, precision: Precision) -> Callable[[NDArray[np.uint8]], NDArray[np.float32]]:
    """Return a function mapping a letterboxed/cropped uint8 HWC image to a
    normalized float32 HWC tensor, per the resolved normalization mode.
    """
    mode = resolve_normalization_mode(input_spec, precision)

    if mode == "none":
        return lambda img: img.astype(np.float32)

    if mode == "unit_float":
        return lambda img: (img.astype(np.float32) / 255.0)

    if mode == "imagenet_bgr":
        mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
        std = np.array(_IMAGENET_STD, dtype=np.float32)
        return lambda img: (img.astype(np.float32) - mean) / std

    if mode == "imagenet_rgb":
        mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
        std = np.array(_IMAGENET_STD, dtype=np.float32)
        return lambda img: (img.astype(np.float32) - mean) / std

    if isinstance(mode, CustomNormalization):
        scale = mode.scale
        mean = np.array(mode.mean or [0.0, 0.0, 0.0], dtype=np.float32)
        std = np.array(mode.std or [1.0, 1.0, 1.0], dtype=np.float32)
        return lambda img: (img.astype(np.float32) * scale - mean) / std

    raise ValueError(f"Unsupported normalization mode: {mode!r}")


def sidecar_model_spec(
    sidecar: SidecarModel,
    *,
    size: str,
    batch_key: str,
    precision: Precision,
    sidecar_dir: Path,
    prepare: Callable[[Path], Path] | None = None,
    coreml_options: dict | None = None,
) -> OnnxModelSpec:
    """Build the `OnnxModelSpec` for one (size, batch, precision) artifact.

    Resolves the artifact's `url`/`filename`/`url_sha256` through the sidecar's
    own directory (`sidecar_dir`), per specs/sidecar-spec.md "Storage layout" —
    the sidecar and its ONNX files live together in one leaf directory.
    """
    resolved_size = sidecar.resolved_size(size)
    group = resolved_size.onnx.batch_artifacts[batch_key]
    artifact = group.precision_artifacts[precision]

    if artifact.url is not None:
        source = ModelSource(url=artifact.url)
    else:
        source = ModelSource(local_path=str(sidecar_dir / artifact.filename))

    # Force resolution now so callers get a concrete, cached local path;
    # OnnxModelSpec.source stays a ModelSource for OnnxSession.create() to
    # resolve again (idempotent — already cached after this call).
    resolve_model_path(
        source,
        cache_dir=sidecar_dir,
        expected_filename=artifact.filename,
        expected_sha256=artifact.url_sha256,
    )

    target_size = resolved_size.input.resize.target_size if resolved_size.input.resize else None
    input_size = tuple(target_size) if target_size is not None else tuple(resolved_size.input.shape[2:4])

    return OnnxModelSpec(
        name=f"{sidecar.model_id}-{size}",
        source=ModelSource(local_path=str(sidecar_dir / artifact.filename)),
        input_size=input_size,  # type: ignore[arg-type]
        prepare=prepare,
        coreml_options=coreml_options,
    )
