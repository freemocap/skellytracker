"""Generic image preprocessing (resize + normalize) for detector inference.

Free functions any detector's `preprocess()` can call directly.
`preprocess_image` is driven by an `InputSpec` — the model I/O contract a
sidecar YAML describes — but that's just where the config value comes from;
this module is about processing images, not about the sidecar format, so it
lives with the other detector code rather than under `core/sidecar/`.

Only the resize variants exercised by a real model today are implemented;
everything else the spec permits raises `NotImplementedError` rather than
silently mis-processing the image.
"""

from __future__ import annotations

import cv2
import numpy as np
from beartype.typing import Callable
from numpy.typing import NDArray

from skellytracker.core.sidecar.model import CustomNormalization, InputSpec, Precision

_CV2_INTERPOLATION: dict[str, int] = {
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
}

_IMAGENET_MEAN: tuple[float, float, float] = (123.675, 116.28, 103.53)
_IMAGENET_STD: tuple[float, float, float] = (58.395, 57.12, 57.375)


# ==========================================================================
# Normalization
# ==========================================================================


def resolve_normalization_mode(
    input_spec: InputSpec, precision: Precision
) -> str | CustomNormalization:
    """Resolve the effective normalization mode for `precision`.

    Resolution order (see specs/sidecar-spec.md, "Normalization modes"):
    `normalization_by_precision[precision]` -> top-level `normalization` -> default `imagenet_bgr`.
    """
    by_precision = input_spec.normalization_by_precision or {}
    if precision in by_precision:
        return by_precision[precision]
    return input_spec.normalization


def build_normalization_fn(
    input_spec: InputSpec, precision: Precision
) -> Callable[[NDArray[np.uint8]], NDArray[np.float32]]:
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


# ==========================================================================
# Resize
# ==========================================================================


def letterbox_preprocess(
    img: NDArray[np.uint8],
    target_size: tuple[int, int],
    pad_value: int = 114,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[NDArray, float]:
    """Resize to fit `target_size` preserving aspect ratio, pad with `pad_value`.

    Returns `(padded_img, ratio)` — uint8 HWC, no normalisation applied.
    `ratio` is the single uniform scale factor applied to both axes; divide
    decoded box/point coordinates by it to map back to the original image.
    """
    th, tw = target_size
    padded_img = np.full((th, tw, 3), pad_value, dtype=np.uint8)

    ratio = min(th / img.shape[0], tw / img.shape[1])
    nw, nh = int(img.shape[1] * ratio), int(img.shape[0] * ratio)
    resized = cv2.resize(img, (nw, nh), interpolation=interpolation).astype(np.uint8)
    padded_img[:nh, :nw] = resized

    return padded_img, ratio


def preprocess_image(
    image: NDArray[np.uint8],
    target_size: tuple[int, int],
    input_spec: InputSpec,
    precision: Precision = "fp32",
) -> tuple[NDArray[np.float32], float]:
    """Resize and normalize `image` per `input_spec`.

    Dispatches on `input_spec.resize.method` and the resize sub-options the
    spec allows. New model families fail loudly (`NotImplementedError`) until
    this function grows support for their resize method, rather than being
    silently mis-processed.
    """
    resize = input_spec.resize
    if resize is None:
        raise NotImplementedError(
            "preprocess_image requires input.resize to be declared"
        )

    if resize.method == "letterbox":
        if resize.preserve_aspect_ratio is False:
            raise NotImplementedError(
                "resize.method='letterbox' with preserve_aspect_ratio=false is "
                "not yet implemented — it produces non-uniform x/y scale "
                "factors that the decode side doesn't yet accept in place of "
                "a single scalar `ratio`"
            )
        pad_value = 114 if resize.pad_value is None else resize.pad_value
        interpolation = _CV2_INTERPOLATION[resize.interpolation]
        padded, ratio = letterbox_preprocess(
            image, target_size, pad_value=pad_value, interpolation=interpolation
        )
    elif resize.method == "affine_person_crop":
        raise NotImplementedError(
            "resize.method='affine_person_crop' is not yet implemented"
        )
    elif resize.method == "none":
        raise NotImplementedError("resize.method='none' is not yet implemented")
    else:
        raise NotImplementedError(f"Unknown resize.method: {resize.method!r}")

    normalize = build_normalization_fn(input_spec, precision)
    normalized = normalize(padded)
    tensor = (
        normalized if input_spec.layout == "NHWC" else normalized.transpose(2, 0, 1)
    )
    return np.ascontiguousarray(tensor.astype(np.float32)), ratio
