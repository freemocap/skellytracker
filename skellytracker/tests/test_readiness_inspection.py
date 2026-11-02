"""CPU-only cache evidence tests; no native models or user cache fixtures."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from skellytracker.core.sessions.readiness_inspection import (
    CacheEvidenceState,
    RuntimeVersions,
    inspect_trt_cache,
)

VERSIONS = RuntimeVersions(ort_version="test-ort", trt_version="test-trt")


def write_manifest(directory: Path, **changes) -> None:
    values = {"ort_version": VERSIONS.ort_version, "trt_version": VERSIONS.trt_version}
    values.update(changes)
    (directory / "cache_manifest.json").write_text(json.dumps(values), encoding="utf-8")


def snapshot(directory: Path):
    return {path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in directory.iterdir() if path.is_file()}


def test_absent_cache_is_not_created(tmp_path):
    directory = tmp_path / "absent"
    result = inspect_trt_cache(directory, expected_versions=VERSIONS)
    assert result.state == CacheEvidenceState.MISSING
    assert not directory.exists()
    assert not result.ready


def test_regular_file_is_not_a_cache_directory(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"unchanged")
    assert inspect_trt_cache(path).state == CacheEvidenceState.UNKNOWN
    assert path.read_bytes() == b"unchanged"


@pytest.mark.parametrize("payload", [None, b"broken", b"[]", b"{}", b"\xff",
    b'{"ort_version":1,"trt_version":"test-trt"}',
    b'{"ort_version":"unavailable","trt_version":"test-trt"}',
    b'{"ort_version":"test-ort","trt_version":"test-trt","extra":true}', b"x" * 65537,
    b"[" * 4096 + b"0" + b"]" * 4096],
    ids=["absent", "invalid-json", "list", "empty", "invalid-encoding", "wrong-type",
         "unavailable", "extra-fields", "oversized", "excessive-nesting"])
def test_missing_or_unrecognized_manifest_never_certifies_engines(tmp_path, payload):
    (tmp_path / "some.engine").write_bytes(b"opaque-engine")
    if payload is not None:
        (tmp_path / "cache_manifest.json").write_bytes(payload)
    before = snapshot(tmp_path)
    result = inspect_trt_cache(tmp_path, expected_versions=VERSIONS)
    assert result.state == CacheEvidenceState.UNKNOWN
    assert not result.ready
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("field", ["ort_version", "trt_version"])
def test_runtime_mismatch_preserves_engine_timing_and_manifest(tmp_path, field):
    write_manifest(tmp_path, **{field: "different"})
    (tmp_path / "model.engine").write_bytes(b"opaque-engine")
    (tmp_path / "model.timing").write_bytes(b"timing-data")
    before = snapshot(tmp_path)
    result = inspect_trt_cache(tmp_path, expected_versions=VERSIONS)
    assert result.state == CacheEvidenceState.RUNTIME_MISMATCH
    assert not result.ready
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("batch_label", ["batch_1", "batch_4", "batch_128"])
def test_directory_capacity_and_matching_runtime_are_only_candidates(tmp_path, batch_label):
    directory = tmp_path / batch_label
    directory.mkdir()
    write_manifest(directory)
    engine = directory / "unknown-model.engine"
    engine.write_bytes(b"not-even-a-validated-engine")
    result = inspect_trt_cache(directory, expected_versions=VERSIONS)
    assert result.state == CacheEvidenceState.CANDIDATE
    assert result.engine_files == (engine,)
    assert not result.ready
    assert "compatibility" in result.reason


def test_runtime_unknown_does_not_probe_environment(tmp_path):
    write_manifest(tmp_path)
    (tmp_path / "model.engine").write_bytes(b"engine")
    result = inspect_trt_cache(tmp_path)
    assert result.state == CacheEvidenceState.UNKNOWN
    assert result.recorded_versions == VERSIONS


def test_manifest_without_engines_is_missing_not_ready(tmp_path):
    write_manifest(tmp_path)
    assert inspect_trt_cache(tmp_path, expected_versions=VERSIONS).state == CacheEvidenceState.MISSING


def test_empty_engine_is_not_a_candidate(tmp_path):
    write_manifest(tmp_path)
    (tmp_path / "empty.engine").write_bytes(b"")
    assert inspect_trt_cache(tmp_path, expected_versions=VERSIONS).state == CacheEvidenceState.UNKNOWN


def test_read_failure_is_explicit_and_not_a_cache_mutation(tmp_path, monkeypatch):
    write_manifest(tmp_path)
    before = snapshot(tmp_path)
    with monkeypatch.context() as patch:
        def denied(*args, **kwargs):
            raise PermissionError("controlled read failure")
        patch.setattr(Path, "open", denied)
        result = inspect_trt_cache(tmp_path, expected_versions=VERSIONS)
    assert result.state == CacheEvidenceState.UNKNOWN
    assert "PermissionError" in result.reason
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("version", ["", " ", "unavailable"])
def test_unknown_expected_version_is_rejected(version):
    with pytest.raises(ValueError):
        RuntimeVersions(ort_version=version, trt_version="known")


def test_normal_package_import_and_inspection_do_not_import_native_inference(tmp_path):
    # A fresh interpreter proves the import boundary without hiding cached imports.
    script = '''
import importlib.abc
import pathlib
import sys
class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'onnxruntime', 'tensorrt', 'torch'}:
            raise AssertionError('Unexpected native inference import: ' + fullname)
sys.meta_path.insert(0, RejectNative())
from skellytracker.core.sessions.readiness_inspection import inspect_trt_cache
result = inspect_trt_cache(pathlib.Path(sys.argv[1]))
assert not result.ready
assert not pathlib.Path(sys.argv[1]).exists()
'''
    # Fixed interpreter and literal test code, argument list without a shell.
    completed = subprocess.run([sys.executable, "-B", "-c", script, str(tmp_path / "missing")],  # noqa: S603
                               cwd=Path(__file__).resolve().parents[2], capture_output=True,
                               text=True, timeout=30, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
