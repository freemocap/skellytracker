from __future__ import annotations

from pathlib import Path

import yaml


def load_point_names(path: Path) -> tuple[str, ...]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return tuple(data["tracked_points"])


def load_connections(path: Path) -> tuple[tuple[str, str], ...]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return tuple((a, b) for a, b in data.get("connections", []))
