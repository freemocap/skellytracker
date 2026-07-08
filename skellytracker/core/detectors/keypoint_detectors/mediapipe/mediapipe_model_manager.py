from __future__ import annotations

import hashlib
import logging
from enum import Enum
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".freemocap" / "skellytracker-models"


class MediapipePoseModelComplexity(str, Enum):
    LITE = "lite"
    FULL = "full"
    HEAVY = "heavy"


POSE_MODEL_URLS: dict[MediapipePoseModelComplexity, str] = {
    MediapipePoseModelComplexity.LITE: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    MediapipePoseModelComplexity.FULL: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    MediapipePoseModelComplexity.HEAVY: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def _download_model(url: str, target_path: Path) -> None:
    logger.info(f"Downloading model from {url} to {target_path}...")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url=url, stream=True, timeout=120)
    response.raise_for_status()

    tmp_path = target_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        tmp_path.rename(target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    logger.info(f"Model downloaded successfully: {target_path}")


def _filename_from_url(url: str) -> str:
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
    base_name = url.rsplit("/", maxsplit=1)[-1]
    return f"{url_hash}_{base_name}"


def get_model_path(url: str) -> Path:
    """Return local path for a model, downloading it if not already cached."""
    filename = _filename_from_url(url)
    local_path = CACHE_DIR / filename
    if not local_path.exists():
        _download_model(url=url, target_path=local_path)
    return local_path


def get_pose_model_path(complexity: MediapipePoseModelComplexity) -> Path:
    return get_model_path(url=POSE_MODEL_URLS[complexity])


def get_hand_model_path() -> Path:
    return get_model_path(url=HAND_MODEL_URL)


def get_face_model_path() -> Path:
    return get_model_path(url=FACE_MODEL_URL)
