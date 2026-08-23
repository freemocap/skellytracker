"""Tests for the sidecar-oriented additions to model_registry.resolve_model_path:
`expected_filename` (name override) and `expected_sha256` (integrity check).
No real network access — `requests.get` is mocked.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from skellytracker.core.sessions.model_registry import ModelIntegrityError, ModelSource, resolve_model_path


def _fake_response(body: bytes) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.headers = {"content-length": str(len(body))}
    resp.iter_content = MagicMock(return_value=[body])
    return resp


class TestExpectedFilename:
    def test_onnx_download_uses_expected_filename_not_url_tail(self, tmp_path: Path):
        body = b"fake-onnx-bytes"
        with patch("skellytracker.core.sessions.model_registry.requests.get", return_value=_fake_response(body)):
            path = resolve_model_path(
                ModelSource(url="https://example.com/upstream-name.onnx"),
                cache_dir=tmp_path,
                expected_filename="sidecar-declared-name.onnx",
            )
        assert path.name == "sidecar-declared-name.onnx"
        assert path.read_bytes() == body

    def test_zip_extraction_uses_expected_filename(self, tmp_path: Path):
        import zipfile

        onnx_bytes = b"fake-onnx-inside-zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("upstream_internal_name.onnx", onnx_bytes)
        zip_bytes = buf.getvalue()

        with patch("skellytracker.core.sessions.model_registry.requests.get", return_value=_fake_response(zip_bytes)):
            path = resolve_model_path(
                ModelSource(url="https://example.com/checkpoint.zip"),
                cache_dir=tmp_path,
                expected_filename="sidecar-declared-name.onnx",
            )
        assert path.name == "sidecar-declared-name.onnx"
        assert path.read_bytes() == onnx_bytes


class TestExpectedSha256:
    def test_matching_sha256_succeeds(self, tmp_path: Path):
        body = b"fake-onnx-bytes"
        digest = hashlib.sha256(body).hexdigest()
        with patch("skellytracker.core.sessions.model_registry.requests.get", return_value=_fake_response(body)):
            path = resolve_model_path(
                ModelSource(url="https://example.com/model.onnx"),
                cache_dir=tmp_path,
                expected_sha256=digest,
            )
        assert path.exists()

    def test_mismatched_sha256_raises_and_does_not_cache(self, tmp_path: Path):
        body = b"fake-onnx-bytes"
        wrong_digest = "0" * 64
        with patch("skellytracker.core.sessions.model_registry.requests.get", return_value=_fake_response(body)):
            with pytest.raises(ModelIntegrityError):
                resolve_model_path(
                    ModelSource(url="https://example.com/model.onnx"),
                    cache_dir=tmp_path,
                    expected_sha256=wrong_digest,
                )
        assert not (tmp_path / "model.onnx").exists()

    def test_sha256_check_skipped_when_not_provided(self, tmp_path: Path):
        body = b"fake-onnx-bytes"
        with patch("skellytracker.core.sessions.model_registry.requests.get", return_value=_fake_response(body)):
            path = resolve_model_path(ModelSource(url="https://example.com/model.onnx"), cache_dir=tmp_path)
        assert path.exists()

    def test_local_path_source_ignores_expected_filename_and_sha256(self, tmp_path: Path):
        local_file = tmp_path / "local.onnx"
        local_file.write_bytes(b"local-bytes")
        path = resolve_model_path(
            ModelSource(local_path=str(local_file)),
            expected_filename="ignored.onnx",
            expected_sha256="ignored",
        )
        assert path == local_file
