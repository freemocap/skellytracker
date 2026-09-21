"""Shared `prefixes` + `[prefix]` expansion for tracker-mapping YAML.

Used by both `TrackerMapping` (`canonical_mapping`/`derived_points` YAML files)
and the sidecar `SidecarModel` (`pose.canonical_mapping`/`pose.derived_points`
fields) — kept dependency-free so neither has to import the other's package.

See specs/sidecar-spec.md, "pose.canonical_mapping" / "pose.derived_points".
"""
from __future__ import annotations

from typing import Any

_PREFIX_TOKEN = "[prefix]"  # noqa: S105 (not a credential, just the expansion marker token)


def expand_prefixed_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    """Expand a `prefixes`-authored mapping dict into its flat, literal form.

    `raw` may contain an optional `prefixes: list[str]` key. Any other key
    starting with `[prefix]` is expanded once per declared prefix, applying
    the prefix to both the canonical/derived name and every source name in
    its entry (string/list/dict forms). Keys without `[prefix]` pass through
    unchanged. Returns a new dict; `raw` is not mutated.
    """
    raw = dict(raw)
    prefixes = raw.pop("prefixes", None)

    result: dict[str, Any] = {}
    for key, entry in raw.items():
        if isinstance(key, str) and key.startswith(_PREFIX_TOKEN):
            if not prefixes:
                raise ValueError(
                    f"Entry {key!r} uses `[prefix]` expansion but no `prefixes` list was declared"
                )
            suffix = key[len(_PREFIX_TOKEN):]
            for prefix in prefixes:
                result[f"{prefix}{suffix}"] = _apply_prefix_to_entry(entry, prefix)
        else:
            result[key] = entry
    return result


def _apply_prefix_to_entry(entry: Any, prefix: str) -> Any:
    if isinstance(entry, str):
        return f"{prefix}{entry}"
    if isinstance(entry, list):
        return [f"{prefix}{name}" for name in entry]
    if isinstance(entry, dict):
        return {f"{prefix}{name}": weight for name, weight in entry.items()}
    raise TypeError(f"Mapping entry must be str, list, or dict, got {type(entry).__name__}")
