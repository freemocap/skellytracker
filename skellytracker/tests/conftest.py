from __future__ import annotations

import hashlib
import logging
import pathlib
import zipfile

import cv2
import numpy as np
import pytest
import requests

logger = logging.getLogger(__name__)

_TEST_IMAGE_URL = "https://github.com/freemocap/skellysamples/releases/download/single_images_v07_12_26/sample_recording_a_pose_image.jpg"
_CHARUCO_TEST_IMAGE_URL = "https://github.com/freemocap/skellysamples/releases/download/single_images_v07_12_26/sample_recording_charuco_image.jpg"
_TEST_RECORDING_URL = "https://github.com/freemocap/skellysamples/releases/download/test_data_v06_09_25/freemocap_test_data.zip"
_MAX_RETRIES = 3

_CACHE_DIR = pathlib.Path.home() / ".cache" / "skellytracker" / "test_images"
_FREEMOCAP_CANONICAL_SYNC_DIR = (
    pathlib.Path.home() / "freemocap_data" / "recordings" / "freemocap_test_data" / "synchronized_videos"
)
_VIDEO_CACHE_DIR = pathlib.Path.home() / ".cache" / "skellytracker" / "test_data"


class _SessionInfo:
    test_image: np.ndarray | None = None
    charuco_test_image: np.ndarray | None = None
    download_error: str | None = None
    charuco_download_error: str | None = None
    sync_videos_dir: pathlib.Path | None = None
    video_error: str | None = None


def _cache_path(url: str) -> pathlib.Path:
    url_hash = hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()[:12]
    return _CACHE_DIR / f"{url_hash}.jpg"


def _load_or_download_image(url: str) -> np.ndarray | None:
    path = _cache_path(url)
    if path.exists():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            logger.debug(f"Loaded test image from cache: {path}")
            return image
        logger.warning(f"Cached file {path} could not be decoded; re-downloading.")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(_MAX_RETRIES):
        try:
            r = requests.get(url, timeout=(5, 60), allow_redirects=True)
            r.raise_for_status()
            if len(r.content) == 0:
                logger.warning(f"Empty response on attempt {attempt + 1} (status={r.status_code})")
                continue
            arr = np.frombuffer(r.content, np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is not None:
                cv2.imwrite(str(path), image)
                logger.debug(f"Downloaded and cached test image to {path}")
                return image
            logger.warning(f"cv2.imdecode returned None on attempt {attempt + 1}")
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
    return None


def _get_or_download_test_recording() -> pathlib.Path | None:
    if _FREEMOCAP_CANONICAL_SYNC_DIR.exists():
        mp4s = list(_FREEMOCAP_CANONICAL_SYNC_DIR.glob("*.mp4"))
        if mp4s:
            logger.debug(f"Using existing test recording at {_FREEMOCAP_CANONICAL_SYNC_DIR}")
            return _FREEMOCAP_CANONICAL_SYNC_DIR

    extracted_dir = _VIDEO_CACHE_DIR / "freemocap_test_data" / "synchronized_videos"
    if extracted_dir.exists():
        mp4s = list(extracted_dir.glob("*.mp4"))
        if mp4s:
            logger.debug(f"Using cached test recording at {extracted_dir}")
            return extracted_dir

    _VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _VIDEO_CACHE_DIR / "freemocap_test_data.zip"

    for attempt in range(_MAX_RETRIES):
        try:
            r = requests.get(_TEST_RECORDING_URL, timeout=(10, 300), allow_redirects=True, stream=True)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            break
        except Exception as e:
            logger.warning(f"Recording download attempt {attempt + 1} failed: {e}")
            if zip_path.exists():
                zip_path.unlink()
    else:
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(_VIDEO_CACHE_DIR)
    except zipfile.BadZipFile as e:
        logger.warning(f"Downloaded zip is corrupt: {e}")
        zip_path.unlink(missing_ok=True)
        return None

    candidates = [
        c for c in _VIDEO_CACHE_DIR.rglob("synchronized_videos")
        if c.is_dir() and list(c.glob("*.mp4"))
    ]
    if not candidates:
        logger.warning("Extracted zip does not contain a synchronized_videos directory with .mp4 files")
        return None

    return candidates[0]


def pytest_configure(config: pytest.Config) -> None:
    np.set_printoptions(threshold=20, edgeitems=3)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--fail-on-skip",
        action="store_true",
        default=False,
        help="Fail (rather than skip) any test that would otherwise be skipped.",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo
) -> pytest.Generator:
    outcome = yield
    report = outcome.get_result()
    if report.skipped and item.config.getoption("--fail-on-skip", default=False):
        report.outcome = "failed"
        report.longrepr = f"[--fail-on-skip] {report.longrepr}"


def pytest_sessionstart(session: pytest.Session) -> None:
    image = _load_or_download_image(_TEST_IMAGE_URL)
    if image is None:
        _SessionInfo.download_error = f"Could not load test image from {_TEST_IMAGE_URL} after {_MAX_RETRIES} attempts"
        logger.warning(_SessionInfo.download_error)
    else:
        _SessionInfo.test_image = image

    charuco_image = _load_or_download_image(_CHARUCO_TEST_IMAGE_URL)
    if charuco_image is None:
        _SessionInfo.charuco_download_error = f"Could not load charuco test image from {_CHARUCO_TEST_IMAGE_URL} after {_MAX_RETRIES} attempts"
        logger.warning(_SessionInfo.charuco_download_error)
    else:
        _SessionInfo.charuco_test_image = charuco_image

    result = _get_or_download_test_recording()
    if result is None:
        _SessionInfo.video_error = (
            f"Could not load test recording. Checked {_FREEMOCAP_CANONICAL_SYNC_DIR} "
            f"and tried downloading from {_TEST_RECORDING_URL}."
        )
        logger.warning(_SessionInfo.video_error)
    else:
        _SessionInfo.sync_videos_dir = result


@pytest.fixture()
def test_image() -> np.ndarray:
    if _SessionInfo.download_error is not None:
        pytest.skip(_SessionInfo.download_error)
    return _SessionInfo.test_image


@pytest.fixture()
def charuco_test_image() -> np.ndarray:
    if _SessionInfo.charuco_download_error is not None:
        pytest.skip(_SessionInfo.charuco_download_error)
    return _SessionInfo.charuco_test_image


@pytest.fixture(scope="session")
def sync_videos_dir() -> pathlib.Path:
    if _SessionInfo.video_error is not None:
        pytest.skip(_SessionInfo.video_error)
    return _SessionInfo.sync_videos_dir


@pytest.fixture(scope="session")
def test_video_path(sync_videos_dir: pathlib.Path) -> pathlib.Path:
    """First .mp4 (alphabetically) from the test recording."""
    mp4s = sorted(sync_videos_dir.glob("*.mp4"))
    if not mp4s:
        pytest.skip(f"No .mp4 files found in {sync_videos_dir}")
    return mp4s[0]
