from __future__ import annotations

from abc import ABC

from pydantic import BaseModel


class ObjectDetectorConfig(BaseModel, ABC):
    """Base config for ObjectDetector implementations.

    Subclasses set detector_type to a Literal value and add model-specific fields.
    session_backend must match a key in the sessions dict passed to create().
    """

    detector_type: str
    session_backend: str


class KeypointDetectorConfig(BaseModel, ABC):
    """Base config for KeypointDetector implementations.

    Subclasses set detector_type to a Literal value and add model-specific fields.
    session_backend must match a key in the sessions dict passed to create().
    """

    detector_type: str
    session_backend: str
