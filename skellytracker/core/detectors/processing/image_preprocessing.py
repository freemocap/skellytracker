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
from skellytracker.core.detectors.metadata import ImageTransform

from skellytracker.core.detectors.processing.affine_crop import (
    bbox_xyxy2cs,
    top_down_affine,
)
from skellytracker.core.sidecar.model import (
    CropPolicySpec,
    CustomNormalization,
    InputSpec,
    Precision,
)

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
    if input_spec.dtype.get(precision) != "float32":
        raise NotImplementedError(
            "Image preprocessing currently requires declared float32 inputs"
        )
    mode = resolve_normalization_mode(input_spec, precision)

    if mode == "none":
        return lambda img: img.astype(np.float32)

    if mode == "unit_float":
        return lambda img: img.astype(np.float32) / 255.0

    if mode == "imagenet_bgr":
        mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
        std = np.array(_IMAGENET_STD, dtype=np.float32)
        return lambda img: (img.astype(np.float32) - mean) / std

    if mode == "imagenet_rgb":
        mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
        std = np.array(_IMAGENET_STD, dtype=np.float32)
        return lambda img: (img[..., ::-1].astype(np.float32) - mean) / std

    if isinstance(mode, CustomNormalization):
        scale = mode.scale
        mean = np.array(mode.mean or [0.0, 0.0, 0.0], dtype=np.float32)
        std = np.array(mode.std or [1.0, 1.0, 1.0], dtype=np.float32)
        if mode.color_format == "rgb":
            return lambda img: (img[..., ::-1].astype(np.float32) * scale - mean) / std
        return lambda img: (img.astype(np.float32) * scale - mean) / std

    raise ValueError(f"Unsupported normalization mode: {mode!r}")


# ==========================================================================
# Resize
# ==========================================================================


def letterbox_preprocess(
    img: NDArray[np.uint8],
    target_size: tuple[int, int],
    pad_value: int | float | tuple[int | float, ...] | list[int | float] = 114,
    interpolation: int = cv2.INTER_LINEAR,
    *,
    pad_position: str | None = None,
    rounding: str | None = None,
) -> tuple[NDArray, float | ImageTransform]:
    """Resize to fit `target_size` preserving aspect ratio, pad with `pad_value`.

    Returns `(padded_img, ratio)` — uint8 HWC, no normalisation applied.
    `ratio` is the single uniform scale factor applied to both axes; divide
    decoded box/point coordinates by it to map back to the original image.
    """
    th, tw = target_size
    padded_img = np.full((th, tw, 3), pad_value, dtype=np.uint8)

    ratio = min(th / img.shape[0], tw / img.shape[1])
    if rounding not in (None, "half_even", "half_up"):
        raise ValueError(f"Unknown resize rounding: {rounding}")
    if pad_position not in (None, "top_left", "center"):
        raise ValueError(f"Unknown padding position: {pad_position}")
    round_size = (
        int
        if rounding is None
        else round
        if rounding == "half_even"
        else lambda value: int(np.floor(value + 0.5))
    )
    nw = min(tw, max(1, round_size(img.shape[1] * ratio)))
    nh = min(th, max(1, round_size(img.shape[0] * ratio)))
    resized = cv2.resize(img, (nw, nh), interpolation=interpolation).astype(np.uint8)
    left, top = ((tw - nw) // 2, (th - nh) // 2) if pad_position == "center" else (0, 0)
    padded_img[top : top + nh, left : left + nw] = resized
    if pad_position is not None or rounding is not None:
        return padded_img, ImageTransform(
            nw / img.shape[1], nh / img.shape[0], float(left), float(top)
        )
    return padded_img, ratio


def fit_preprocess(
    img: NDArray[np.uint8],
    target_size: tuple[int, int],
    pad_value: int | float | tuple[int | float, ...] | list[int | float] = 0,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[NDArray[np.uint8], tuple[float, float, float]]:
    """Full-frame affine fit resize centered on the target canvas.

    Returns `(padded_img, (scale, left, top))` where `scale` is the uniform
    affine scale factor and `(left, top)` are the fractional canvas offsets
    used by the warp matrix.
    """
    target_h, target_w = target_size
    src_h, src_w = img.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    left = (target_w - src_w * scale) / 2.0
    top = (target_h - src_h * scale) / 2.0

    M = np.array([[scale, 0.0, left], [0.0, scale, top]], dtype=np.float64)
    if isinstance(pad_value, (list, tuple, np.ndarray)):
        border_value = tuple(float(v) for v in pad_value)
    else:
        border_value = (float(pad_value),) * img.shape[2]

    padded_img = cv2.warpAffine(
        img,
        M,
        (target_w, target_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return padded_img, (float(scale), float(left), float(top))


def preprocess_image(
    image: NDArray[np.uint8],
    target_size: tuple[int, int],
    input_spec: InputSpec,
    precision: Precision = "fp32",
    bbox_xyxy: NDArray[np.floating] | None = None,
    normalization_fn: Callable[[NDArray[np.uint8]], NDArray[np.float32]] | None = None,
) -> tuple[
    NDArray[np.float32],
    float | ImageTransform | tuple[NDArray[np.float64], NDArray[np.float64]],
]:
    """Resize and normalize `image` per `input_spec`.

    Dispatches on `input_spec.resize.method` and the resize sub-options the
    spec allows. New model families fail loudly (`NotImplementedError`) until
    this function grows support for their resize method, rather than being
    silently mis-processed.

    Returns `(tensor, resize_meta)`. Existing letterbox declarations without
    explicit padding/rounding retain their scalar ratio. Explicit letterbox,
    stretch and fit return ImageTransform with the actual scales and offsets.
    Affine person crops return `(center, scale)` arrays for SIMCC decoding.

    `bbox_xyxy` is only used by `affine_person_crop`; `None` crops the full
    image (today's only usage — no upstream-detector bbox is wired through
    yet).
    """
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("Image preprocessing requires a nonempty BGR image")
    if min(target_size) < 1:
        raise ValueError("Target dimensions must be positive")
    if input_spec.dtype.get(precision) != "float32":
        raise NotImplementedError(
            "Image preprocessing currently requires declared float32 inputs"
        )
    resize = input_spec.resize
    if resize is None:
        raise NotImplementedError(
            "preprocess_image requires input.resize to be declared"
        )

    resize_meta: (
        float | ImageTransform | tuple[NDArray[np.float64], NDArray[np.float64]]
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
        padded, resize_meta = letterbox_preprocess(
            image,
            target_size,
            pad_value=pad_value,
            interpolation=interpolation,
            pad_position=resize.pad_position,
            rounding=resize.rounding,
        )
    elif resize.method == "stretch":
        interpolation = _CV2_INTERPOLATION[resize.interpolation]
        target_h, target_w = target_size
        padded = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        resize_meta = ImageTransform(
            target_w / image.shape[1], target_h / image.shape[0]
        )
    elif resize.method == "fit":
        pad_value = 0 if resize.pad_value is None else resize.pad_value
        interpolation = _CV2_INTERPOLATION[resize.interpolation]
        padded, (fit_scale, fit_left, fit_top) = fit_preprocess(
            image, target_size, pad_value=pad_value, interpolation=interpolation
        )
        resize_meta = ImageTransform(fit_scale, fit_scale, fit_left, fit_top)
    elif resize.method == "affine_person_crop":
        crop_policy = resize.crop_policy or CropPolicySpec()
        if not crop_policy.maintain_aspect_ratio:
            raise NotImplementedError(
                "resize.crop_policy.maintain_aspect_ratio=false is not yet implemented"
            )
        h, w = image.shape[:2]
        bbox = (
            bbox_xyxy
            if bbox_xyxy is not None
            else np.array([0.0, 0.0, float(w), float(h)], dtype=np.float64)
        )
        center, scale = bbox_xyxy2cs(bbox, padding=crop_policy.expand_ratio)
        target_h, target_w = target_size
        padded, scale = top_down_affine(
            (target_w, target_h),
            scale,
            center,
            image,
            interpolation=_CV2_INTERPOLATION[resize.interpolation],
        )
        resize_meta = (center, scale)
    elif resize.method == "none":
        raise NotImplementedError("resize.method='none' is not yet implemented")
    else:
        raise NotImplementedError(f"Unknown resize.method: {resize.method!r}")

    normalize = normalization_fn or build_normalization_fn(input_spec, precision)
    normalized = normalize(padded)
    tensor = (
        normalized if input_spec.layout == "NHWC" else normalized.transpose(2, 0, 1)
    )
    return np.ascontiguousarray(tensor, dtype=np.float32), resize_meta
