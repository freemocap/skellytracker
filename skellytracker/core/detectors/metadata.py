from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class YoloxMetadata:
    ratio: float


@dataclass
class RTMPoseMetadata:
    center: NDArray[np.float64]  # shape (2,)
    scale: NDArray[np.float64]  # shape (2,)


@dataclass
class EmptyMetadata:
    pass
