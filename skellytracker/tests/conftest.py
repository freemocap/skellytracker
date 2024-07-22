import numpy as np
import pytest
from skellytracker.system.default_paths import FIGSHARE_CHARUCO_TEST_IMAGE_URL
from skellytracker.utilities.download_test_image import download_test_image


class SessionInfo:
    test_image: np.ndarray
    charuco_test_image: np.ndarray


def pytest_sessionstart(session):
    SessionInfo.test_image = download_test_image()
    SessionInfo.charuco_test_image = download_test_image(test_image_url=FIGSHARE_CHARUCO_TEST_IMAGE_URL)


@pytest.fixture()
def test_image():
    return SessionInfo.test_image

@pytest.fixture
def charuco_test_image():
    return SessionInfo.charuco_test_image
