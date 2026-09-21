"""Tests for skellytracker.core.sidecar.resolution — pure $ref/base/deep-merge logic.

Fixture files are written to `tmp_path` per test (acting as `{cache_dir}`)
rather than checked into the repo, so each test's directory tree is visible
right next to the assertions it backs.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from skellytracker.core.sidecar.errors import SidecarParseError, SidecarRefError
from skellytracker.core.sidecar.resolution import _deep_merge, resolve_sidecar_composition


def _write_yaml(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh)


# ---------------------------------------------------------------------------
# _deep_merge (JSON Merge Patch semantics)
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_mapping_plus_mapping_merges_recursively(self):
        base = {"a": 1, "nested": {"x": 1, "y": 2}}
        override = {"nested": {"y": 20, "z": 3}}
        assert _deep_merge(base, override) == {"a": 1, "nested": {"x": 1, "y": 20, "z": 3}}

    def test_null_deletes_key(self):
        base = {"a": 1, "b": 2}
        override = {"b": None}
        assert _deep_merge(base, override) == {"a": 1}

    def test_sequence_replaces_wholesale_not_concatenated(self):
        base = {"list": [1, 2, 3]}
        override = {"list": [4]}
        assert _deep_merge(base, override) == {"list": [4]}

    def test_mapping_replaced_by_scalar(self):
        base = {"a": {"x": 1}}
        override = {"a": 5}
        assert _deep_merge(base, override) == {"a": 5}

    def test_key_absent_in_base_is_added(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# $ref resolution
# ---------------------------------------------------------------------------


class TestRefResolution:
    def test_ref_replaces_node_with_fragment_content(self, tmp_path: Path):
        _write_yaml(tmp_path / "fragment.yaml", ["a", "b", "c"])
        _write_yaml(tmp_path / "main.yaml", {"tracked_points": {"$ref": "fragment.yaml"}})

        result = resolve_sidecar_composition(tmp_path / "main.yaml", tmp_path)
        assert result == {"tracked_points": ["a", "b", "c"]}

    def test_ref_path_resolves_relative_to_containing_file(self, tmp_path: Path):
        _write_yaml(tmp_path / "shared" / "fragment.yaml", {"foo": "bar"})
        _write_yaml(tmp_path / "nested" / "dir" / "main.yaml", {"x": {"$ref": "../../shared/fragment.yaml"}})

        result = resolve_sidecar_composition(tmp_path / "nested" / "dir" / "main.yaml", tmp_path)
        assert result == {"x": {"foo": "bar"}}

    def test_nested_ref_resolves_relative_to_its_own_file(self, tmp_path: Path):
        _write_yaml(tmp_path / "a" / "leaf.yaml", {"value": 1})
        _write_yaml(tmp_path / "b" / "middle.yaml", {"inner": {"$ref": "../a/leaf.yaml"}})
        _write_yaml(tmp_path / "main.yaml", {"outer": {"$ref": "b/middle.yaml"}})

        result = resolve_sidecar_composition(tmp_path / "main.yaml", tmp_path)
        assert result == {"outer": {"inner": {"value": 1}}}

    def test_ref_with_sibling_keys_is_an_error(self, tmp_path: Path):
        _write_yaml(tmp_path / "fragment.yaml", {"foo": "bar"})
        _write_yaml(tmp_path / "main.yaml", {"x": {"$ref": "fragment.yaml", "other": "y"}})

        with pytest.raises(SidecarRefError):
            resolve_sidecar_composition(tmp_path / "main.yaml", tmp_path)

    def test_missing_ref_target_is_an_error(self, tmp_path: Path):
        _write_yaml(tmp_path / "main.yaml", {"x": {"$ref": "does_not_exist.yaml"}})

        with pytest.raises(SidecarRefError):
            resolve_sidecar_composition(tmp_path / "main.yaml", tmp_path)

    def test_ref_cycle_is_detected(self, tmp_path: Path):
        _write_yaml(tmp_path / "a.yaml", {"x": {"$ref": "b.yaml"}})
        _write_yaml(tmp_path / "b.yaml", {"y": {"$ref": "a.yaml"}})

        with pytest.raises(SidecarRefError):
            resolve_sidecar_composition(tmp_path / "a.yaml", tmp_path)

    def test_ref_escaping_cache_dir_is_rejected(self, tmp_path: Path):
        cache_dir = tmp_path / "cache_root"
        _write_yaml(tmp_path / "outside.yaml", {"foo": "bar"})
        _write_yaml(cache_dir / "main.yaml", {"x": {"$ref": "../outside.yaml"}})

        with pytest.raises(SidecarRefError):
            resolve_sidecar_composition(cache_dir / "main.yaml", cache_dir)

    def test_malformed_yaml_raises_parse_error(self, tmp_path: Path):
        path = tmp_path / "broken.yaml"
        path.write_text("key: [unclosed")

        with pytest.raises(SidecarParseError):
            resolve_sidecar_composition(path, tmp_path)


# ---------------------------------------------------------------------------
# base resolution
# ---------------------------------------------------------------------------


class TestBaseResolution:
    def test_base_deep_merges_current_over_base(self, tmp_path: Path):
        _write_yaml(tmp_path / "base.yaml", {"model_id": "base", "sizes": {"nano": {"a": 1, "b": 2}}})
        _write_yaml(
            tmp_path / "derived.yaml",
            {"base": "base.yaml", "model_id": "derived", "sizes": {"nano": {"b": 20}}},
        )

        result = resolve_sidecar_composition(tmp_path / "derived.yaml", tmp_path)
        assert result == {"model_id": "derived", "sizes": {"nano": {"a": 1, "b": 20}}}

    def test_base_null_delete_removes_inherited_key(self, tmp_path: Path):
        _write_yaml(tmp_path / "base.yaml", {"sizes": {"nano": {"int8": {"filename": "x.onnx"}, "fp32": {"filename": "y.onnx"}}}})
        _write_yaml(tmp_path / "derived.yaml", {"base": "base.yaml", "sizes": {"nano": {"int8": None}}})

        result = resolve_sidecar_composition(tmp_path / "derived.yaml", tmp_path)
        assert result == {"sizes": {"nano": {"fp32": {"filename": "y.onnx"}}}}

    def test_base_chain_resolves_depth_first(self, tmp_path: Path):
        _write_yaml(tmp_path / "grandparent.yaml", {"a": 1, "b": 1})
        _write_yaml(tmp_path / "parent.yaml", {"base": "grandparent.yaml", "b": 2, "c": 2})
        _write_yaml(tmp_path / "child.yaml", {"base": "parent.yaml", "c": 3, "d": 3})

        result = resolve_sidecar_composition(tmp_path / "child.yaml", tmp_path)
        assert result == {"a": 1, "b": 2, "c": 3, "d": 3}

    def test_ref_inside_base_resolves_relative_to_base_file(self, tmp_path: Path):
        _write_yaml(tmp_path / "base_dir" / "fragment.yaml", ["p1", "p2"])
        _write_yaml(tmp_path / "base_dir" / "base.yaml", {"points": {"$ref": "fragment.yaml"}})
        _write_yaml(tmp_path / "derived_dir" / "derived.yaml", {"base": "../base_dir/base.yaml"})

        result = resolve_sidecar_composition(tmp_path / "derived_dir" / "derived.yaml", tmp_path)
        assert result == {"points": ["p1", "p2"]}

    def test_missing_base_target_is_an_error(self, tmp_path: Path):
        _write_yaml(tmp_path / "derived.yaml", {"base": "missing.yaml"})

        with pytest.raises(SidecarRefError):
            resolve_sidecar_composition(tmp_path / "derived.yaml", tmp_path)
