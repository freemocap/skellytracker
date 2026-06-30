from __future__ import annotations

from abc import ABC

from pydantic import BaseModel


class SessionConfig(BaseModel, ABC):
    """Base config for Session implementations.

    Subclasses add backend-specific fields and set backend to a Literal value.
    """

    backend: str
