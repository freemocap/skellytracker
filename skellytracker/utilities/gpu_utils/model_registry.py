"""Generic model resolution and download — framework-agnostic.

A ``ModelSource`` says *where* to get a model file (URL, Hugging Face Hub, or
local path).  ``resolve_model_path()`` returns a local ``Path``, downloading
and caching if necessary.  ``ModelSpec`` bundles a source with the metadata
needed to run inference (input size, keypoint count, preprocessing contract).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile
from enum import Enum
from pathlib import Path
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict


class TrackerPreset(str, Enum):
    """High-level performance tier that bundles model choices for all components."""

    light = "light"
    medium = "medium"
    heavy = "heavy"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default cache directory
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "skellytracker" / "models"


# ---------------------------------------------------------------------------
# Well-known ONNX model URLs (OpenMMLab CDN)
# ---------------------------------------------------------------------------

MODEL_URLS: dict[str, str] = {
    # -- RTMO body (one-stage) --
    "rtmo-s": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
        "rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip"
    ),
    "rtmo-m": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
        "rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip"
    ),
    "rtmo-l": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
        "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip"
    ),
    # -- RTMPose hand --
    "rtmpose-hand": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
    ),
    # -- RTMPose face (LaPa 106-point) --
    "rtmpose-face": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.zip"
    ),
    # -- YOLOX detection --
    "yolox-tiny": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip"
    ),
    "yolox-m": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
    ),
    # -- RTMPose wholebody (SIMCC) --
    "rtmpose-s": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip"
    ),
    "rtmpose-m": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
    ),
    # -- RTMW wholebody (cocktail14, 133 kpt) --
    "rtmw-l-m_256x192": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
        "rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip"
    ),
    "rtmw-x-l_256x192": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
        "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip"
    ),
    "rtmw-x-l_384x288": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
        "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
    ),
    # -- MediaPipe hand landmark (PINTO dynamic-batch ONNX, NCHW, 21 kpt) --
    "mediapipe-hand-landmark": (
        "https://raw.githubusercontent.com/PINTO0309/"
        "hand-gesture-recognition-using-onnx/main/model/"
        "hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx"
    ),
    # -- MediaPipe pose landmark (OpenCV ONNX conversion, 33 keypoints) --
    "mediapipe-pose-landmark": (
        "https://huggingface.co/opencv/pose_estimation_mediapipe/"
        "resolve/main/pose_estimation_mediapipe_2023mar.onnx"
    ),
    # -- MediaPipe palm detection (OpenCV ONNX conversion) --
    "mediapipe-palm-detector": (
        "https://huggingface.co/opencv/palm_detection_mediapipe/"
        "resolve/main/palm_detection_mediapipe_2023feb.onnx"
    ),
    # -- MediaPipe face detection + landmarks (Qualcomm ONNX) --
    "mediapipe-face-detector-short": (
        "https://huggingface.co/qualcomm/MediaPipe-Face-Detection/"
        "resolve/main/MediaPipeFaceDetector.onnx"
    ),
    "mediapipe-face-landmark": (
        "https://huggingface.co/qualcomm/MediaPipe-Face-Detection/"
        "resolve/main/MediaPipeFaceLandmarkDetector.onnx"
    ),
}


# ---------------------------------------------------------------------------
# Model source descriptor
# ---------------------------------------------------------------------------


class ModelSource(BaseModel):
    """Where to get a model file.  Set exactly one of the optional fields.

    When *local_path* is set, no download is performed — the path is returned
    as-is (wrapped in ``Path``).
    """

    model_config = ConfigDict(frozen=True)

    url: str | None = None
    """Direct download URL.  For OpenMMLab CDN this is a ``.zip`` containing
    a single ``.onnx`` file."""

    hf_repo: str | None = None
    """Hugging Face Hub repository ID (e.g. ``"JunkyByte/easy_ViTPose"``)."""

    hf_filename: str | None = None
    """File path within the HF repo (e.g. ``"torch/wholebody/vitpose-h-wholebody.pth"``)."""

    local_path: str | None = None
    """Absolute or relative path to an already-downloaded model file."""


# ---------------------------------------------------------------------------
# Model spec (framework-agnostic)
# ---------------------------------------------------------------------------


class ModelSpec(BaseModel):
    """Descriptor for a single ML model used by a tracker.

    Framework-agnostic — works for ONNX, PyTorch, TorchScript, TensorRT, etc.
    """

    model_config = ConfigDict(frozen=True)

    # -- identity ----------------------------------------------------------

    source: ModelSource
    """Where to obtain the model file."""

    format: Literal["onnx", "pth", "pt", "engine"] = "onnx"
    """Model file format.  Informs downstream code which inference backend to use."""

    # -- input / output contract -------------------------------------------

    input_size: tuple[int, int] = (256, 256)
    """Model input tensor spatial dimensions ``(height, width)``."""

    num_keypoints: int = 21
    """Number of keypoints the model outputs per instance."""

    # -- preprocessing contract --------------------------------------------

    preprocess_mode: Literal[
        "rtmo", "rtmpose_letterbox", "simple_letterbox", "mediapipe", "none"
    ] = "simple_letterbox"
    """Which preprocessing pipeline to use.

    ``"mediapipe"`` — RGB conversion + [0,1] scaling + resize to input_size.
    Used by MediaPipe-derived models (hand landmark, pose landmark, etc.).
    Outputs are direct coordinate regression (not SIMCC / heatmaps).
    """

    mean: tuple[float, float, float] | None = None
    """BGR channel-wise mean for normalization.  ``None`` = skip."""

    std: tuple[float, float, float] | None = None
    """BGR channel-wise std for normalization.  ``None`` = skip."""

    # -- SIMCC models only -------------------------------------------------

    simcc_split_ratio: float | None = None
    """SIMCC label resolution divisor.  Only set for SIMCC-based pose models."""

    # -- batch support ------------------------------------------------------

    supports_batching: bool | None = None
    """Whether the model supports batched (N > 1) inference.

    - ``None`` (default): probe at runtime via ``probe_supports_batch()``.
    - ``True``: known to support batching.
    - ``False``: known to NOT support batching — the inference pipeline
      should fall back to sequential per-crop inference and log a warning.
    """

    # ======================================================================
    # Convenience constructors
    # ======================================================================

    # -- Body (RTMO one-stage) ---------------------------------------------

    @classmethod
    def rtmo_light(cls) -> "ModelSpec":
        return cls(
            source=ModelSource(url=MODEL_URLS["rtmo-s"]),
            input_size=(640, 640),
            num_keypoints=17,
            preprocess_mode="rtmo",
        )

    @classmethod
    def rtmo_medium(cls) -> "ModelSpec":
        return cls(
            source=ModelSource(url=MODEL_URLS["rtmo-m"]),
            input_size=(640, 640),
            num_keypoints=17,
            preprocess_mode="rtmo",
        )

    @classmethod
    def rtmo_heavy(cls) -> "ModelSpec":
        return cls(
            source=ModelSource(url=MODEL_URLS["rtmo-l"]),
            input_size=(640, 640),
            num_keypoints=17,
            preprocess_mode="rtmo",
        )

    # -- Hand (RTMPose SIMCC) ----------------------------------------------

    @classmethod
    def rtmpose_hand(cls) -> "ModelSpec":
        return cls(
            source=ModelSource(url=MODEL_URLS["rtmpose-hand"]),
            input_size=(256, 256),
            num_keypoints=21,
            preprocess_mode="rtmpose_letterbox",
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            simcc_split_ratio=2.0,
        )

    # -- Face (RTMPose SIMCC, LaPa 106-point) ------------------------------

    @classmethod
    def rtmpose_face(cls) -> "ModelSpec":
        return cls(
            source=ModelSource(url=MODEL_URLS["rtmpose-face"]),
            input_size=(256, 256),
            num_keypoints=106,
            preprocess_mode="rtmpose_letterbox",
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            simcc_split_ratio=2.0,
        )

    # -- MediaPipe hand landmark (OpenCV ONNX, 21 keypoints, 224×224) --------

    @classmethod
    def mediapipe_hand_landmark(cls) -> "ModelSpec":
        """MediaPipe hand landmark model converted to ONNX by OpenCV Zoo.

        Estimates 21 hand keypoints (x, y, z) from a 224×224 RGB hand crop.
        Input: float32 RGB [0, 1].  Output: direct coordinate regression.
        """
        return cls(
            source=ModelSource(url=MODEL_URLS["mediapipe-hand-landmark"]),
            input_size=(224, 224),
            num_keypoints=21,
            preprocess_mode="mediapipe",
        )

    # -- MediaPipe pose landmark (OpenCV ONNX, 33 keypoints) -----------------

    @classmethod
    def mediapipe_pose_landmark(cls) -> "ModelSpec":
        """MediaPipe pose landmark model converted to ONNX by OpenCV Zoo.

        Estimates 33 body keypoints + segmentation mask from a person crop.
        Input: float32 RGB [0, 1].  Output: direct coordinate regression.
        Requires a person detector (e.g. ``mediapipe_palm_detector``) upstream.
        """
        return cls(
            source=ModelSource(url=MODEL_URLS["mediapipe-pose-landmark"]),
            input_size=(256, 256),
            num_keypoints=33,
            preprocess_mode="mediapipe",
        )

    # -- MediaPipe palm detector (OpenCV ONNX) --------------------------------

    @classmethod
    def mediapipe_palm_detector(cls) -> "ModelSpec":
        """MediaPipe BlazePalm palm detection model converted to ONNX by OpenCV Zoo.

        Detects palm bounding boxes and 7 palm keypoints from a full image.
        Used as the upstream detector for hand landmark models.
        """
        return cls(
            source=ModelSource(url=MODEL_URLS["mediapipe-palm-detector"]),
            input_size=(192, 192),
            num_keypoints=7,
            preprocess_mode="mediapipe",
        )

    # -- MediaPipe face detector (Qualcomm ONNX, short-range) -----------------

    @classmethod
    def mediapipe_face_detector_short(cls) -> "ModelSpec":
        """MediaPipe BlazeFace short-range detector converted to ONNX by Qualcomm.

        Detects face bounding boxes from a full image.  Designed for
        selfie-range faces (within ~2m).
        """
        return cls(
            source=ModelSource(url=MODEL_URLS["mediapipe-face-detector-short"]),
            input_size=(128, 128),
            num_keypoints=6,
            preprocess_mode="mediapipe",
        )

    # -- MediaPipe face landmark (Qualcomm ONNX) -------------------------------

    @classmethod
    def mediapipe_face_landmark(cls) -> "ModelSpec":
        """MediaPipe face landmark model converted to ONNX by Qualcomm.

        Estimates 6 facial landmarks (eyes, nose, mouth corners, ear tragions)
        from a face crop.  Input: float32 RGB [0, 1].

        Note: this is the *sparse* 6-point model, not the full MediaPipe
        Face Mesh (468 points).  The full face mesh ONNX is available from
        PINTO model zoo (model 030_BlazeFace).
        """
        return cls(
            source=ModelSource(url=MODEL_URLS["mediapipe-face-landmark"]),
            input_size=(192, 192),
            num_keypoints=6,
            preprocess_mode="mediapipe",
        )


# ==========================================================================
# Resolution
# ==========================================================================


def resolve_model_path(
    source: ModelSource,
    cache_dir: Path | str | None = None,
) -> Path:
    """Return the local filesystem path to a model, downloading if necessary.

    Parameters
    ----------
    source : ModelSource
        Where to obtain the model.  If *local_path* is set it is returned
        directly; otherwise the model is downloaded (and cached) from the
        specified URL or Hugging Face repo.
    cache_dir : Path | str | None
        Directory for cached downloads.  Defaults to
        ``~/.cache/skellytracker/models/``.

    Returns
    -------
    Path
        Absolute path to the model file on disk.
    """
    if source.local_path:
        path = Path(source.local_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Local model not found: {path}")
        return path

    if source.hf_repo is not None:
        return _resolve_from_huggingface(source, cache_dir)

    if source.url is not None:
        return _resolve_from_url(source.url, cache_dir)

    raise ValueError("ModelSource must specify one of: local_path, hf_repo, url")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_cache() -> Path:
    d = DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_from_huggingface(
    source: ModelSource,
    cache_dir: Path | str | None = None,
) -> Path:
    """Download a file from Hugging Face Hub.

    Uses ``huggingface_hub.hf_hub_download`` so that the HF cache is
    respected, avoiding re-downloads.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download models from Hugging Face. "
            "Install it with: pip install huggingface_hub"
        )

    if source.hf_filename is None:
        raise ValueError("hf_filename is required for Hugging Face downloads")

    path = hf_hub_download(
        repo_id=source.hf_repo,
        filename=source.hf_filename,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    logger.info(f"Model from HF: {path}")
    return Path(path)


def _resolve_from_url(
    url: str,
    cache_dir: Path | str | None = None,
) -> Path:
    """Download a model from a URL, caching the result.

    Handles two URL conventions:

    - **OpenMMLab CDN**: the URL points to a ``.zip`` containing one or more
      ``.onnx`` files.  The first ``.onnx`` found is cached by its expected
      filename.
    - **Direct .onnx URL**: the URL points directly to an ``.onnx`` file
      (e.g. Hugging Face resolve links).  The file is downloaded and cached
      as-is.
    """
    cache = Path(cache_dir) if cache_dir else _default_cache()
    filename = url.rsplit("/", 1)[-1]

    # --- Direct .onnx download (Hugging Face, etc.) ---
    if filename.endswith(".onnx"):
        cached_onnx = cache / filename
        if cached_onnx.exists():
            logger.info(f"Using cached model: {cached_onnx}")
            return cached_onnx

        logger.info(f"Downloading model from {url} ...")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        # Download to a temp file, then atomically move to cache.
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
        try:
            total = 0
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
                total += len(chunk)
            tmp.close()
            logger.info(f"Downloaded {total / 1024 / 1024:.1f} MiB")
            shutil.move(tmp.name, str(cached_onnx))
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

        logger.info(f"Model cached: {cached_onnx}")
        return cached_onnx

    # --- OpenMMLab CDN .zip convention ---
    # Derive the expected ONNX filename from the URL stem.
    # e.g.  ".../rtmo-m_..._20231211.zip"  →  "rtmo-m_..._20231211.onnx"
    onnx_name = filename.replace(".zip", ".onnx")
    cached_onnx = cache / onnx_name

    if cached_onnx.exists():
        logger.info(f"Using cached model: {cached_onnx}")
        return cached_onnx

    # Download the zip to a temp file.
    logger.info(f"Downloading model from {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        total = 0
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
            total += len(chunk)
        tmp.close()
        logger.info(f"Downloaded {total / 1024 / 1024:.1f} MiB")

        # Extract the first .onnx from the zip, then rename to the
        # expected cache name.  (OpenMMLab zips consistently name the
        # ONNX "end2end.onnx"; using the URL-derived name per model
        # prevents one model overwriting another.)
        with zipfile.ZipFile(tmp.name, "r") as zf:
            onnx_names = [n for n in zf.namelist() if n.endswith(".onnx")]
            if not onnx_names:
                raise RuntimeError(f"No .onnx found in zip: {url}")
            with zf.open(onnx_names[0]) as src:
                cached_onnx.write_bytes(src.read())
    finally:
        os.unlink(tmp.name)

    logger.info(f"Model cached: {cached_onnx}")
    return cached_onnx
