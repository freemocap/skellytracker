from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np

from skellytracker.core.data_primitives.observation import Observation


@dataclass
class Annotator(ABC):
    """Draws detection results onto an image.

    Reads point names and skeleton connections from YAML-defined schemas.
    Mostly detector-agnostic — works from any Observation.
    """

    @abstractmethod
    def annotate(
        self,
        image: NDArray[np.uint8],
        observation: Observation,
    ) -> NDArray[np.uint8]:
        """Return the image with bounding boxes, keypoints, and connections drawn."""
        ...

    @classmethod
    @abstractmethod
    def create(cls, config: object) -> Annotator:
        ...
