import logging

import cv2
import numpy as np
import requests

from skellytracker.system.default_paths import FIGSHARE_TEST_IMAGE_URL


logger = logging.getLogger(__name__)


def download_test_image(test_image_url: str = FIGSHARE_TEST_IMAGE_URL) -> np.ndarray:
    try:
        logger.info(f"Downloading test image from {test_image_url}...")

        r = requests.get(test_image_url, stream=True, timeout=(5, 60))
        r.raise_for_status()  # Check if request was successful

        image_array = np.frombuffer(r.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        logger.info("Test image downloaded successfully.")
        return image

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise e


if __name__ == "__main__":
    test_data_path = download_test_image()
