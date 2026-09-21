"""skellytracker version parsing/comparison for sidecar `schema_version` gating.

See specs/sidecar-spec.md, "Schema versioning".
"""
from __future__ import annotations

import re

from skellytracker.core.sidecar.errors import SidecarSchemaVersionError

_VERSION_RE = re.compile(r"^v(\d{4})\.(\d{2})\.(\d+)(?:-([A-Za-z0-9.]+))?$")


def parse_skellytracker_version(version: str) -> tuple[int, int, int, str | None]:
    """Parse `vYYYY.0M.BUILD[-TAG]` into `(year, month, build, tag)`."""
    match = _VERSION_RE.match(version)
    if match is None:
        raise SidecarSchemaVersionError(
            f"{version!r} does not match the skellytracker version pattern vYYYY.0M.BUILD[-TAG]"
        )
    year, month, build, tag = match.groups()
    return int(year), int(month), int(build), tag


def _sort_key(version: str) -> tuple[int, int, int, tuple[int, str]]:
    year, month, build, tag = parse_skellytracker_version(version)
    # Stable (tag=None) sorts after a tagged pre-release of the same core.
    tag_key = (1, "") if tag is None else (0, tag)
    return (year, month, build, tag_key)


def is_schema_version_supported(sidecar_version: str, installed_version: str) -> bool:
    """`True` when `installed_version >= sidecar_version` (skellytracker can load the sidecar)."""
    return _sort_key(installed_version) >= _sort_key(sidecar_version)


def require_stable_version(version: str) -> None:
    """Raise if `version` carries a pre-release `-TAG` suffix (sidecars must be stable)."""
    _, _, _, tag = parse_skellytracker_version(version)
    if tag is not None:
        raise SidecarSchemaVersionError(
            f"schema_version {version!r} carries a pre-release tag; sidecars must use stable versions only"
        )
