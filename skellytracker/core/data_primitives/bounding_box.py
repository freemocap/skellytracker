from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class BoundingBox:
    """Rectangular region in pixel space produced by an ObjectDetector."""

    x1: int | float
    y1: int | float
    x2: int | float
    y2: int | float
    confidence: int | float = 1.0

    def __post_init__(self) -> None:
        self.x1 = float(self.x1)
        self.y1 = float(self.y1)
        self.x2 = float(self.x2)
        self.y2 = float(self.y2)
        self.confidence = float(self.confidence)
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                f"Invalid BoundingBox: (x1={self.x1}, y1={self.y1}) must be <= "
                f"(x2={self.x2}, y2={self.y2})"
            )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def size(self) -> tuple[float, float]:
        return self.width, self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def diagonal(self) -> float:
        return float(np.sqrt(self.width**2 + self.height**2))

    # ------------------------------------------------------------------
    # Modification
    # ------------------------------------------------------------------

    def scaled(self, factor: float) -> BoundingBox:
        """Expand (or contract) the box around its center by a scale factor."""
        cx, cy = self.center
        hw = self.width * factor / 2.0
        hh = self.height * factor / 2.0
        return BoundingBox(
            x1=cx - hw, y1=cy - hh,
            x2=cx + hw, y2=cy + hh,
            confidence=self.confidence,
        )

    def padded(self, px: float) -> BoundingBox:
        """Expand the box by a fixed number of pixels on each side."""
        return BoundingBox(
            x1=self.x1 - px, y1=self.y1 - px,
            x2=self.x2 + px, y2=self.y2 + px,
            confidence=self.confidence,
        )

    def clipped(self, image_height: int, image_width: int) -> BoundingBox:
        """Clip the box to lie within image bounds."""
        return BoundingBox(
            x1=max(0.0, self.x1),
            y1=max(0.0, self.y1),
            x2=min(float(image_width), self.x2),
            y2=min(float(image_height), self.y2),
            confidence=self.confidence,
        )

    # ------------------------------------------------------------------
    # Cropping
    # ------------------------------------------------------------------

    def to_crop(self, image: NDArray) -> NDArray:
        """Return the image region bounded by this box."""
        y1 = max(0, int(self.y1))
        y2 = max(0, int(self.y2))
        x1 = max(0, int(self.x1))
        x2 = max(0, int(self.x2))
        return image[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def full_image(image_height: int, image_width: int) -> BoundingBox:
        """A box spanning the entire image; used when no ObjectDetector is present."""
        return BoundingBox(
            x1=0.0, y1=0.0,
            x2=float(image_width), y2=float(image_height),
            confidence=1.0,
        )

    @staticmethod
    def from_center_size(
        cx: float, cy: float, width: float, height: float, confidence: float = 1.0
    ) -> BoundingBox:
        return BoundingBox(
            x1=cx - width / 2.0,
            y1=cy - height / 2.0,
            x2=cx + width / 2.0,
            y2=cy + height / 2.0,
            confidence=confidence,
        )
