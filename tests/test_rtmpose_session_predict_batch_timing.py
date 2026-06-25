"""GPU test: predict_batch populates stage timings and ordered task events."""
from pathlib import Path

import cv2
import pytest

from skellytracker.trackers.base_tracker.task_events import (
    NODE_KIND_SKELETON_INFERENCE,
    RTMPOSE_BATCH_STAGES,
    TrackerTaskEventCollector,
    make_batch_task_id,
    make_camera_task_id,
)
from skellytracker.trackers.rtmpose_tracker.rtmpose_session import (
    RTMPoseSession,
    RTMPoseSessionConfig,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "000000406129.jpg"


def _assert_six_stage_timings(session: RTMPoseSession) -> None:
    assert session.last_human_detection_preprocess_ms > 0.0
    assert session.last_human_detection_ms > 0.0
    assert session.last_human_detection_postprocess_ms > 0.0
    assert session.last_pose_estimation_preprocess_ms > 0.0
    assert session.last_pose_estimation_ms > 0.0
    assert session.last_pose_estimation_postprocess_ms > 0.0


def _assert_ordered_task_events(
    collector: TrackerTaskEventCollector,
    *,
    frame_number: int,
    camera_ids: list[str] | None,
    parent_task_ids: tuple[str, ...],
) -> None:
    events = collector.events
    assert len(events) == len(RTMPOSE_BATCH_STAGES)

    stage_names = [event.stage for event in events]
    assert stage_names == list(RTMPOSE_BATCH_STAGES)

    for index, event in enumerate(events):
        assert event.frame_number == frame_number
        assert event.node_kind == NODE_KIND_SKELETON_INFERENCE
        assert event.parent_task_ids == parent_task_ids
        assert event.duration_ms > 0.0
        assert event.end_time_ns > event.start_time_ns
        assert abs(event.duration_ms - (event.end_time_ns - event.start_time_ns) / 1e6) < 0.01

        if event.camera_id is not None:
            expected_id = make_camera_task_id(
                frame_number=frame_number,
                camera_id=event.camera_id,
                node_kind=NODE_KIND_SKELETON_INFERENCE,
                stage=event.stage,
            )
        else:
            expected_id = make_batch_task_id(
                frame_number=frame_number,
                node_kind=NODE_KIND_SKELETON_INFERENCE,
                stage=event.stage,
            )
        assert event.task_id == expected_id

    for earlier, later in zip(events, events[1:]):
        assert later.start_time_ns >= earlier.start_time_ns

    if camera_ids is not None:
        human_detection_stages = set(RTMPOSE_BATCH_STAGES[:3])
        for event in events:
            if event.stage in human_detection_stages:
                assert event.camera_id == camera_ids[0]
            else:
                assert event.camera_id is None


@pytest.mark.gpu
def test_predict_batch_exposes_six_stage_timings() -> None:
    image = cv2.imread(str(_FIXTURE))
    assert image is not None, f"Missing fixture {_FIXTURE}"

    session = RTMPoseSession.create(
        RTMPoseSessionConfig(mode="lightweight", execution_provider="cuda", max_batch_size=1),
    )
    session.predict_batch([image])

    _assert_six_stage_timings(session)


@pytest.mark.gpu
def test_predict_batch_emits_ordered_task_events_with_context() -> None:
    image = cv2.imread(str(_FIXTURE))
    assert image is not None, f"Missing fixture {_FIXTURE}"

    frame_number = 123
    camera_ids = ["webcam_0"]
    parent_task_ids = ["122:webcam_0:camera:human_detection"]
    collector = TrackerTaskEventCollector()

    session = RTMPoseSession.create(
        RTMPoseSessionConfig(mode="lightweight", execution_provider="cuda", max_batch_size=1),
    )
    session.predict_batch(
        [image],
        frame_number=frame_number,
        camera_ids=camera_ids,
        parent_task_ids=parent_task_ids,
        event_collector=collector,
    )

    _assert_six_stage_timings(session)
    _assert_ordered_task_events(
        collector,
        frame_number=frame_number,
        camera_ids=camera_ids,
        parent_task_ids=tuple(parent_task_ids),
    )
