"""Generic model resolution and download — framework-agnostic.

A ``ModelSource`` says *where* to get a model file (URL, Hugging Face Hub, or
local path).  ``resolve_model_path()`` returns a local ``Path``, downloading
and caching if necessary.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm


logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "skellytracker" / "models"


class ModelIntegrityError(Exception):
    """Raised when a downloaded model's bytes do not match its expected SHA-256."""


class ModelSource(BaseModel):
    """Where to get a model file.  Set exactly one of the optional fields."""

    model_config = ConfigDict(frozen=True)

    url: str | None = None
    hf_repo: str | None = None
    hf_filename: str | None = None
    local_path: str | None = None


def resolve_model_path(
    source: ModelSource,
    cache_dir: Path | str | None = None,
    expected_filename: str | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Return the local filesystem path to a model, downloading if necessary.

    `expected_filename`/`expected_sha256` let a sidecar-driven caller pin the
    cached file's name (rather than deriving it from the URL tail) and verify
    its integrity against the sidecar's declared `url_sha256` — see
    specs/sidecar-spec.md, "sizes.<size>.onnx.batch_artifacts". Both are
    ignored for `local_path` sources (already-resolved local files are not
    re-verified).
    """
    if source.local_path:
        path = Path(source.local_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Local model not found: {path}")
        return path

    if source.hf_repo is not None:
        return _resolve_from_huggingface(source, cache_dir)

    if source.url is not None:
        return _resolve_from_url(
            source.url, cache_dir, expected_filename=expected_filename, expected_sha256=expected_sha256
        )

    raise ValueError("ModelSource must specify one of: local_path, hf_repo, url")


def _default_cache() -> Path:
    d = DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_from_huggingface(
    source: ModelSource,
    cache_dir: Path | str | None = None,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download models from Hugging Face. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    if source.hf_filename is None:
        raise ValueError("hf_filename is required for Hugging Face downloads")

    path = hf_hub_download(
        repo_id=source.hf_repo,
        filename=source.hf_filename,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    logger.info(f"Model from HF: {path}")
    return Path(path)


def _download_to_temp(url: str, *, suffix: str, expected_sha256: str | None, filename_for_progress: str) -> str:
    """Download `url` to a temp file, verify its bytes against `expected_sha256` if given.

    Returns the temp file path. Caller owns cleanup of the returned path.
    """
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    hasher = hashlib.sha256() if expected_sha256 else None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        with tqdm(
            total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
            desc=filename_for_progress, miniters=1,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
                if hasher is not None:
                    hasher.update(chunk)
                pbar.update(len(chunk))
        tmp.close()
    except Exception:
        os.unlink(tmp.name)
        raise

    if expected_sha256 is not None:
        actual = hasher.hexdigest()
        if actual.lower() != expected_sha256.lower():
            os.unlink(tmp.name)
            raise ModelIntegrityError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256.lower()}, got {actual}"
            )

    return tmp.name


def _resolve_from_url(
    url: str,
    cache_dir: Path | str | None = None,
    expected_filename: str | None = None,
    expected_sha256: str | None = None,
) -> Path:
    cache = Path(cache_dir) if cache_dir else _default_cache()
    filename = url.rsplit("/", 1)[-1]

    if filename.endswith(".onnx"):
        cached_onnx = cache / (expected_filename or filename)
        if cached_onnx.exists():
            logger.info(f"Using cached model: {cached_onnx}")
            return cached_onnx

        logger.info(f"Downloading model from {url} ...")
        tmp_path = _download_to_temp(url, suffix=".tmp", expected_sha256=expected_sha256, filename_for_progress=filename)
        try:
            cache.mkdir(parents=True, exist_ok=True)
            shutil.move(tmp_path, str(cached_onnx))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        logger.info(f"Model cached: {cached_onnx}")
        return cached_onnx

    onnx_name = expected_filename or filename.replace(".zip", ".onnx")
    cached_onnx = cache / onnx_name

    if cached_onnx.exists():
        logger.info(f"Using cached model: {cached_onnx}")
        return cached_onnx

    logger.info(f"Downloading model from {url} ...")
    tmp_path = _download_to_temp(url, suffix=".zip", expected_sha256=expected_sha256, filename_for_progress=filename)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmp_path, "r") as zf:
            onnx_names = [n for n in zf.namelist() if n.endswith(".onnx")]
            if not onnx_names:
                raise RuntimeError(f"No .onnx found in zip: {url}")
            with zf.open(onnx_names[0]) as src:
                cached_onnx.write_bytes(src.read())
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    logger.info(f"Model cached: {cached_onnx}")
    return cached_onnx


def resolve_model_paths_parallel(
    sources: list[ModelSource],
    cache_dir: Path | str | None = None,
) -> dict[int, Path]:
    """Resolve multiple model paths in parallel using a thread pool."""
    if not sources:
        return {}

    results: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(sources))) as pool:
        futures = {
            pool.submit(resolve_model_path, src, cache_dir): i
            for i, src in enumerate(sources)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning(
                    f"Model resolution failed for source {idx} "
                    f"({sources[idx].url or sources[idx].hf_repo or 'local'}): {e!r}"
                )
    return results
