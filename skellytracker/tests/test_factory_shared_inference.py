from unittest.mock import patch

import numpy as np

from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.detectors.keypoint_detectors.rtmpose import (
    RTMPoseDetectorConfig,
)
from skellytracker.core.detectors.object_detectors.yolox import (
    YoloxPersonDetectorConfig,
)
from skellytracker.core.sessions.onnx_session import OnnxSession
from skellytracker.core.sessions.shared_sessions import SharedSessionPool
from skellytracker.core.tracker import tracker_session_requests, TrackerState


def test_two_trackers_share_real_models_until_last_owner_closes(test_image):
    config = TrackerConfig(
        stages=[
            DetectionStageConfig(
                name="body",
                object_detector=YoloxPersonDetectorConfig(),
                keypoint_detectors=[RTMPoseDetectorConfig()],
            )
        ]
    )
    pool = SharedSessionPool()
    try:
        with patch.object(
            OnnxSession, "create", wraps=OnnxSession.create
        ) as create_session:
            first = pool.create_tracker(
                config=config,
                requests=tracker_session_requests(
                    config, batch_size=2, onnx_overrides={"execution_provider": "cpu"}
                ),
            )
            second = pool.create_tracker(
                config=config,
                requests=tracker_session_requests(
                    config, batch_size=2, onnx_overrides={"execution_provider": "cpu"}
                ),
            )
        create_session.assert_called_once()
        model_session = first._entries[0].session
        assert second._entries[0].session is model_session
        assert (
            first.tracker.stages[0].keypoint_detectors[0]
            is not second.tracker.stages[0].keypoint_detectors[0]
        )
        first_observation, first_state = first.tracker.process_image(
            test_image, 0, TrackerState.empty()
        )
        first.close()
        assert len(model_session._sessions) == 2
        second_observation, second_state = second.tracker.process_image(
            test_image, 0, TrackerState.empty()
        )
        assert first_state is not second_state
        first_points = first_observation.stages["body"].keypoints
        second_points = second_observation.stages["body"].keypoints
        assert first_points.names == second_points.names
        np.testing.assert_array_equal(first_points.xyz, second_points.xyz)
        np.testing.assert_array_equal(first_points.visibility, second_points.visibility)
        second.close()
        assert not model_session._sessions
        assert not pool._entries
    finally:
        pool.close()
