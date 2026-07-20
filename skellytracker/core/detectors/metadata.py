from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class YoloxMetadata:
    ratio: float
    original_size: tuple[int, int]  # (H, W)
    is_prenms: bool = False  # True when the model has pre-NMS outputs


@dataclass
class RTMPoseMetadata:
    center: NDArray[np.float64]  # shape (2,)
    scale: NDArray[np.float64]   # shape (2,)


@dataclass
class EmptyMetadata:
    pass
