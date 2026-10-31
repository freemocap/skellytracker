from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ImageTransform:
    """Source-to-model image scale and translation, with vectorized inverse mapping."""

    scale_x: float
    scale_y: float
    offset_x: float = 0.0
    offset_y: float = 0.0

    def restore_points(self, points: NDArray) -> NDArray[np.float64]:
        if points.shape[-1] != 2:
            raise ValueError("Image coordinates require an x/y component axis")
        result = points.astype(np.float64, copy=True)
        result -= (self.offset_x, self.offset_y)
        result /= (self.scale_x, self.scale_y)
        return result

    def restore_boxes(self, boxes: NDArray) -> NDArray[np.float64]:
        if boxes.shape[-1] != 4:
            raise ValueError("Image boxes require x1/y1/x2/y2 components")
        return self.restore_points(boxes.reshape(*boxes.shape[:-1], 2, 2)).reshape(
            boxes.shape
        )


@dataclass
class YoloxMetadata:
    ratio: float | ImageTransform
    original_size: tuple[int, int]  # (H, W)
    is_prenms: bool = False  # True when the model has pre-NMS outputs


@dataclass
class RTMPoseMetadata:
    center: NDArray[np.float64]  # shape (2,)
    scale: NDArray[np.float64]  # shape (2,)


@dataclass
class EmptyMetadata:
    pass
