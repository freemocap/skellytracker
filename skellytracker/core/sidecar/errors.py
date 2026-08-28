"""Recoverable error types for sidecar parsing, composition, and validation.

Every error names the offending file (and directive, where applicable) so a
caller can present an actionable message pointing at the exact YAML that
caused the failure.
"""
from __future__ import annotations

from pathlib import Path


class SidecarError(Exception):
    """Base class for all sidecar-related errors."""

    def __init__(self, message: str, file_path: Path | None = None) -> None:
        self.file_path = file_path
        if file_path is not None:
            message = f"{message} (file: {file_path})"
        super().__init__(message)


class SidecarParseError(SidecarError):
    """A sidecar or fragment file could not be parsed as YAML."""


class SidecarRefError(SidecarError):
    """A `$ref` directive is invalid: missing target, cycle, sibling keys, or
    a path that escapes `{cache_dir}`."""


class SidecarBaseError(SidecarError):
    """A `base` directive is invalid: missing target or a path that escapes
    `{cache_dir}`."""


class SidecarSchemaVersionError(SidecarError):
    """A sidecar's `schema_version` is unsupported by the installed release,
    or does not match the required version pattern."""


class SidecarValidationError(SidecarError):
    """A resolved sidecar document failed Pydantic model validation.

    Wraps the underlying `pydantic.ValidationError` in `__cause__`.
    """
