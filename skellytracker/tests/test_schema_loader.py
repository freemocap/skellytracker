from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

from skellytracker.core.detectors.keypoint_detectors._schema_loader import (
    load_connections,
    load_point_names,
)
from skellytracker.core.io.tracker_mapping import TrackerMapping


def test_schema_loaders_use_utf8_on_a_gbk_default_system(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    schema_path = tmp_path / "模型定义.yaml"
    schema_path.write_text(
        "# 中文注释用于模拟 Windows 简体中文环境\n"
        "tracked_points:\n"
        "  - left_hip\n"
        "  - right_hip\n"
        "connections:\n"
        "  - [left_hip, right_hip]\n",
        encoding="utf-8",
    )

    real_open = builtins.open

    def open_with_gbk_default(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("encoding", "gbk")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_with_gbk_default)

    assert load_point_names(schema_path) == ("left_hip", "right_hip")
    assert load_connections(schema_path) == (("left_hip", "right_hip"),)


def test_tracker_mapping_uses_utf8_on_a_gbk_default_system(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    mapping_path = tmp_path / "映射定义.yaml"
    mapping_path.write_text(
        "# 中文注释用于模拟 Windows 简体中文环境\n"
        "hips_center: [left_hip, right_hip]\n",
        encoding="utf-8",
    )

    real_open = builtins.open

    def open_with_gbk_default(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("encoding", "gbk")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_with_gbk_default)

    mapping = TrackerMapping.from_yaml(mapping_path)

    assert mapping.canonical_names == ["hips_center"]
