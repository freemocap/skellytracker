"""Generic overlay-group resolution: partition a skeleton's edges into
`overlay.groups` per the spec's prefix-matching rule.

Not a sidecar-parsing concern (see image_preprocessing.py/
object_detection_decode.py for the established precedent) — any
sidecar-driven pose detector with `overlay.groups` can use this.
"""

from __future__ import annotations

from skellytracker.core.sidecar.model import OverlaySpec


def resolve_overlay_groups(
    overlay: OverlaySpec,
    edges: tuple[tuple[str, str], ...],
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Partition `edges` into `overlay.groups`.

    Explicit `connections` groups are checked first (edge order-insensitive),
    then `prefix` groups in declared order (first match wins), then any
    remaining edge goes to the one default group (neither `connections` nor
    `prefix`), if declared. Returns `{}` if `overlay.groups` is unset.
    """
    if not overlay.groups:
        return {}

    groups: dict[str, list[tuple[str, str]]] = {name: [] for name in overlay.groups}
    default_name = next(
        (
            name
            for name, g in overlay.groups.items()
            if g.connections is None and g.prefix is None
        ),
        None,
    )
    explicit_groups: dict[str, set[tuple[str, str]]] = {
        name: set(g.connections)
        for name, g in overlay.groups.items()
        if g.connections is not None
    }
    prefix_groups = [
        (name, g.prefix) for name, g in overlay.groups.items() if g.prefix is not None
    ]

    for a, b in edges:
        matched = False
        for name, edge_set in explicit_groups.items():
            if (a, b) in edge_set or (b, a) in edge_set:
                groups[name].append((a, b))
                matched = True
                break
        if matched:
            continue
        for name, prefix in prefix_groups:
            if a.startswith(prefix) or b.startswith(prefix):
                groups[name].append((a, b))
                matched = True
                break
        if not matched and default_name is not None:
            groups[default_name].append((a, b))

    return {name: tuple(edges) for name, edges in groups.items()}
