"""Public entry point: load and validate a sidecar YAML file.

    parse_sidecar_file(path)            # yaml.safe_load -> raw dict
      -> resolve_sidecar_composition()  # resolve $ref includes + base inheritance -> flat dict
      -> SidecarModel.model_validate()  # typed model
"""
from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from skellytracker import __version__ as _INSTALLED_VERSION
from skellytracker.core.sidecar.errors import SidecarSchemaVersionError, SidecarValidationError
from skellytracker.core.sidecar.model import SidecarModel
from skellytracker.core.sidecar.resolution import resolve_sidecar_composition
from skellytracker.core.sidecar.versioning import is_schema_version_supported


def load_sidecar(path: Path, cache_dir: Path | None = None) -> SidecarModel:
    """Load, resolve, and validate a sidecar file at `path`.

    `cache_dir` defaults to the ancestor directory named `core` (the package's
    default `{cache_dir}`, per specs/sidecar-spec.md "Storage layout"). Pass
    an explicit `cache_dir` for test fixtures that live outside the package.
    """
    path = Path(path).resolve()
    if cache_dir is None:
        cache_dir = _default_cache_dir(path)
    else:
        cache_dir = Path(cache_dir).resolve()

    resolved = resolve_sidecar_composition(path, cache_dir)

    try:
        sidecar = SidecarModel.model_validate(resolved)
    except ValidationError as exc:
        raise SidecarValidationError(f"Sidecar failed validation: {exc}", file_path=path) from exc

    expected_basename = f"{sidecar.model_id}.yaml"
    if path.name != expected_basename:
        raise SidecarValidationError(
            f"model_id {sidecar.model_id!r} does not match sidecar basename {path.name!r} "
            f"(expected {expected_basename!r})",
            file_path=path,
        )

    if not is_schema_version_supported(sidecar.schema_version, _INSTALLED_VERSION):
        raise SidecarSchemaVersionError(
            f"Sidecar schema_version {sidecar.schema_version!r} is newer than the installed "
            f"skellytracker version {_INSTALLED_VERSION!r}. Upgrade skellytracker to load this sidecar.",
            file_path=path,
        )

    return sidecar


def _default_cache_dir(path: Path) -> Path:
    for ancestor in path.parents:
        if ancestor.name == "core":
            return ancestor
    raise ValueError(
        f"Could not infer default cache_dir (no ancestor named 'core') for {path}; pass cache_dir explicitly."
    )
