"""Cache publication stays atomic when downloads overlap or fail."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Event
from zipfile import ZipFile

import pytest

from skellytracker.core.sessions import model_registry


class _Response:
    headers = {}

    def __init__(self, payload, fail=False):
        self.payload = payload
        self.fail = fail

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.payload
        if self.fail:
            raise OSError("interrupted download")


@pytest.mark.parametrize("zipped", [False, True])
def test_concurrent_downloads_publish_one_complete_model(tmp_path, monkeypatch, zipped):
    payload = b"complete model contents"
    if zipped:
        stream = BytesIO()
        with ZipFile(stream, "w") as archive:
            archive.writestr("nested/model.onnx", payload)
        downloaded = stream.getvalue()
    else:
        downloaded = payload
    started, release = Event(), Event()
    calls = []

    def get(*args, **kwargs):
        calls.append(args)
        started.set()
        assert release.wait(timeout=5)
        return _Response(downloaded)

    monkeypatch.setattr(model_registry.requests, "get", get)
    suffix = "zip" if zipped else "onnx"
    url = f"https://example.invalid/model.{suffix}"
    destination = tmp_path / "model.onnx"
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(model_registry._resolve_from_url, url, tmp_path)
        try:
            assert started.wait(timeout=5)
            second = pool.submit(model_registry._resolve_from_url, url, tmp_path)
            assert not destination.exists()
        finally:
            release.set()
        assert first.result(timeout=5) == second.result(timeout=5) == destination
    assert len(calls) == 1
    assert destination.read_bytes() == payload
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("suffix", ["onnx", "zip"])
def test_interrupted_download_leaves_no_model_or_temporary_file(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(model_registry.requests, "get", lambda *a, **kw: _Response(b"partial", fail=True))
    with pytest.raises(OSError, match="interrupted"):
        model_registry._resolve_from_url(f"https://example.invalid/model.{suffix}", tmp_path)
    assert not (tmp_path / "model.onnx").exists()
    assert all(path.suffix == ".lock" for path in tmp_path.iterdir())
