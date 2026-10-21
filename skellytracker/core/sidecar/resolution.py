"""Pure YAML composition: `$ref` includes and `base` inheritance.

No Pydantic dependency here — this module only resolves a sidecar file (plus
any fragments/bases it references) into a single flat `dict`/list/scalar tree.
Validation of that resolved tree happens one layer up, in `model.py`/`loader.py`.

Resolution pipeline (see specs/sidecar-spec.md, "Resolution pipeline"):

    parse_sidecar_file(path)            # yaml.safe_load -> raw dict
      -> resolve_sidecar_composition()  # resolve $ref includes + base inheritance -> flat dict
      -> SidecarModel.model_validate()  # typed model (done in loader.py)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from skellytracker.core.sidecar.errors import SidecarBaseError, SidecarParseError, SidecarRefError


def parse_sidecar_file(path: Path) -> Any:
    """Load one YAML file's raw content (`yaml.safe_load`), no composition resolved."""
    try:
        with open(path, "r") as fh:
            return yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise SidecarParseError(f"Failed to parse YAML: {exc}", file_path=path) from exc
    except OSError as exc:
        raise SidecarParseError(f"Failed to read file: {exc}", file_path=path) from exc


def resolve_sidecar_composition(path: Path, cache_dir: Path) -> Any:
    """Resolve a sidecar file's `$ref` includes and `base` inheritance chain.

    Returns the fully flattened raw tree (dict/list/scalar), with no `$ref`
    or `base` directives remaining. `path` must lie within `cache_dir`.
    """
    path = path.resolve()
    cache_dir = cache_dir.resolve()
    _require_within_cache_dir(path, cache_dir, referencing_file=path)
    return _load_and_resolve(path, cache_dir, [])


def _load_and_resolve(path: Path, cache_dir: Path, stack: list[Path]) -> Any:
    if path in stack:
        chain = " -> ".join(str(p) for p in [*stack, path])
        raise SidecarRefError(f"Cycle detected while resolving composition: {chain}", file_path=path)
    if not path.exists():
        raise SidecarRefError(f"Referenced file not found: {path}", file_path=path)

    raw = parse_sidecar_file(path)
    new_stack = [*stack, path]

    if isinstance(raw, dict) and "base" in raw:
        raw = dict(raw)
        base_rel = raw.pop("base")
        if not isinstance(base_rel, str):
            raise SidecarBaseError(f"`base` must be a file path string, got {type(base_rel).__name__}", file_path=path)
        base_path = _resolve_relative_path(path.parent, base_rel, cache_dir, referencing_file=path)
        base_resolved = _load_and_resolve(base_path, cache_dir, new_stack)
        current_resolved = _resolve_refs(raw, path, cache_dir, new_stack)
        return _deep_merge(base_resolved, current_resolved)

    return _resolve_refs(raw, path, cache_dir, new_stack)


def _resolve_refs(node: Any, current_file: Path, cache_dir: Path, stack: list[Path]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            if len(node) != 1:
                extra_keys = sorted(k for k in node if k != "$ref")
                raise SidecarRefError(
                    f"`$ref` mapping must have only the `$ref` key, found extra keys {extra_keys}",
                    file_path=current_file,
                )
            ref_rel = node["$ref"]
            if not isinstance(ref_rel, str):
                raise SidecarRefError(f"`$ref` value must be a file path string, got {type(ref_rel).__name__}", file_path=current_file)
            ref_path = _resolve_relative_path(current_file.parent, ref_rel, cache_dir, referencing_file=current_file)
            return _load_and_resolve(ref_path, cache_dir, stack)
        return {k: _resolve_refs(v, current_file, cache_dir, stack) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(v, current_file, cache_dir, stack) for v in node]
    return node


def _resolve_relative_path(base_dir: Path, rel: str, cache_dir: Path, referencing_file: Path) -> Path:
    target = (base_dir / rel).resolve()
    _require_within_cache_dir(target, cache_dir, referencing_file=referencing_file)
    return target


def _require_within_cache_dir(target: Path, cache_dir: Path, referencing_file: Path) -> None:
    if not target.is_relative_to(cache_dir):
        raise SidecarRefError(
            f"Path {target} escapes cache_dir {cache_dir}",
            file_path=referencing_file,
        )


def _deep_merge(base: Any, override: Any) -> Any:
    """JSON Merge Patch (RFC 7386) semantics: current (override) wins.

    - mapping + mapping: merge key-by-key, recursing into nested mappings.
      A key mapped to `None` in `override` deletes that key from the result.
    - anything else: `override` replaces `base` entirely (sequences are never
      concatenated).
    """
    if isinstance(base, dict) and isinstance(override, dict):
        result = dict(base)
        for key, value in override.items():
            if value is None:
                result.pop(key, None)
            elif key in result:
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    return override
