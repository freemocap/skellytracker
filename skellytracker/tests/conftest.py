import numpy as np
import pytest
from skellytracker.utilities.download_test_image import download_test_image


class SessionInfo:
    test_image: np.ndarray

def pytest_sessionstart(session):
    SessionInfo.test_image = download_test_image()

@pytest.fixture
def test_image():
    return SessionInfo.test_image