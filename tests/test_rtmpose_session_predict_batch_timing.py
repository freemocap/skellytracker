"""GPU test: predict_batch populates all six generic stage timing attrs."""
from pathlib import Path

import cv2
import pytest

from skellytracker.trackers.rtmpose_tracker.rtmpose_session import (
    RTMPoseSession,
    RTMPoseSessionConfig,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "000000406129.jpg"


@pytest.mark.gpu
def test_predict_batch_exposes_six_stage_timings() -> None:
    image = cv2.imread(str(_FIXTURE))
    assert image is not None, f"Missing fixture {_FIXTURE}"

    session = RTMPoseSession.create(
        RTMPoseSessionConfig(mode="lightweight", execution_provider="cuda", max_batch_size=1),
    )
    session.predict_batch([image])

    assert session.last_human_detection_preprocess_ms > 0.0
    assert session.last_human_detection_ms > 0.0
    assert session.last_human_detection_postprocess_ms > 0.0
    assert session.last_pose_estimation_preprocess_ms > 0.0
    assert session.last_pose_estimation_ms > 0.0
    assert session.last_pose_estimation_postprocess_ms > 0.0
