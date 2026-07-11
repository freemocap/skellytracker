from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest
import requests

logger = logging.getLogger(__name__)

_TEST_IMAGE_URL = "https://figshare.com/ndownloader/files/47043898"
_CHARUCO_TEST_IMAGE_URL = "https://figshare.com/ndownloader/files/47127685"
_MAX_RETRIES = 3


class _SessionInfo:
    test_image: np.ndarray | None = None
    charuco_test_image: np.ndarray | None = None
    download_error: str | None = None
    charuco_download_error: str | None = None


def _download_image(url: str) -> np.ndarray | None:
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
                return image
            logger.warning(f"cv2.imdecode returned None on attempt {attempt + 1}")
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
    return None


def pytest_sessionstart(session: pytest.Session) -> None:
    image = _download_image(_TEST_IMAGE_URL)
    if image is None:
        _SessionInfo.download_error = f"Could not download test image from {_TEST_IMAGE_URL} after {_MAX_RETRIES} attempts"
        logger.warning(_SessionInfo.download_error)
    else:
        _SessionInfo.test_image = image

    charuco_image = _download_image(_CHARUCO_TEST_IMAGE_URL)
    if charuco_image is None:
        _SessionInfo.charuco_download_error = f"Could not download charuco test image from {_CHARUCO_TEST_IMAGE_URL} after {_MAX_RETRIES} attempts"
        logger.warning(_SessionInfo.charuco_download_error)
    else:
        _SessionInfo.charuco_test_image = charuco_image


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
