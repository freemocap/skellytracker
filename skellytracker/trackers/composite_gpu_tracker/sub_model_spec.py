"""Model descriptor for body, hand, and face ONNX models.

Each SubModelSpec bundles the ONNX path, input dimensions, keypoint count,
preprocessing contract, and an optional rtmlib preset reference.  The session
and config layer read from the spec so that swapping checkpoints — or entire
model families — does not require changes to the pipeline code.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class TrackerPreset(str, Enum):
    """High-level performance tier that bundles model choices for all components."""

    light = "light"
    medium = "medium"
    heavy = "heavy"


class SubModelSpec(BaseModel):
    """Descriptor for a single sub-model (body, hand, or face)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    onnx_path: str | None = None
    """Explicit ONNX file path.  When ``None`` the session builds the model
    through the *rtmlib* preset referenced by `rtmlib_factory` / `rtmlib_mode`."""

    input_size: tuple[int, int] = (256, 256)
    """Model input tensor spatial dimensions (height, width)."""

    num_keypoints: int = 21
    """Number of keypoints the model outputs per instance."""

    # ------------------------------------------------------------------
    # Preprocessing contract
    # ------------------------------------------------------------------

    preprocess_mode: Literal["rtmo", "rtmpose", "none"] = "rtmpose"
    """Which preprocessing pipeline to use.
    ``"rtmo"`` delegates to rtmlib's ``RTMO.preprocess`` (one-stage body).
    ``"rtmpose"`` uses ``_simple_letterbox`` + the normalisation below.
    ``"none"`` uses ``_simple_letterbox`` without normalisation."""

    mean: tuple[float, float, float] | None = None
    """BGR channel-wise mean subtracted after letterbox.  ``None`` = skip."""

    std: tuple[float, float, float] | None = None
    """BGR channel-wise std divisor applied after letterbox.  ``None`` = skip."""

    simcc_split_ratio: float = 2.0
    """SIMCC label resolution divisor (input_size dim → simcc bins)."""

    # ------------------------------------------------------------------
    # rtmlib auto-download reference (only used when onnx_path is None)
    # ------------------------------------------------------------------

    rtmlib_factory: str | None = None
    """Which rtmlib class / factory to call for auto-download.
    Supported values: ``"rtmo"``, ``"rtmpose_hand"``, ``"rtmpose_face"``."""

    rtmlib_mode: str | None = None
    """Mode string passed to the rtmlib factory (e.g. ``"balanced"``)."""

    # ==================================================================
    # Factory helpers
    # ==================================================================

    # -- body (RTMO one-stage) -----------------------------------------

    @classmethod
    def rtmo_light(cls) -> "SubModelSpec":
        return cls(
            input_size=(640, 640),
            num_keypoints=17,
            preprocess_mode="rtmo",
            rtmlib_factory="rtmo",
            rtmlib_mode="lightweight",
        )

    @classmethod
    def rtmo_medium(cls) -> "SubModelSpec":
        return cls(
            input_size=(640, 640),
            num_keypoints=17,
            preprocess_mode="rtmo",
            rtmlib_factory="rtmo",
            rtmlib_mode="balanced",
        )

    @classmethod
    def rtmo_heavy(cls) -> "SubModelSpec":
        return cls(
            input_size=(640, 640),
            num_keypoints=17,
            preprocess_mode="rtmo",
            rtmlib_factory="rtmo",
            rtmlib_mode="performance",
        )

    # -- hand (RTMPose SIMCC) ------------------------------------------

    @classmethod
    def rtmpose_hand(cls) -> "SubModelSpec":
        return cls(
            input_size=(256, 256),
            num_keypoints=21,
            preprocess_mode="rtmpose",
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            rtmlib_factory="rtmpose_hand",
            rtmlib_mode="lightweight",
        )

    # -- face (RTMPose SIMCC, LaPa 106-point) --------------------------

    @classmethod
    def rtmpose_face(cls) -> "SubModelSpec":
        return cls(
            input_size=(256, 256),
            num_keypoints=106,
            preprocess_mode="rtmpose",
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            rtmlib_factory="rtmpose_face",
            rtmlib_mode=None,
        )
