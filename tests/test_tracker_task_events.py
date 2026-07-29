"""Unit tests for tracker task-event helpers."""
from skellytracker.core.tracker.task_events import (
    NODE_KIND_SKELETON_INFERENCE,
    RTMPOSE_BATCH_STAGES,
    TrackerTaskEventCollector,
    TrackerTaskEventContext,
    make_batch_task_id,
    make_camera_task_id,
    normalize_parent_task_ids,
)


def test_make_batch_task_id_is_deterministic() -> None:
    task_id = make_batch_task_id(
        frame_number=42,
        node_kind=NODE_KIND_SKELETON_INFERENCE,
        stage="human_detection",
    )
    assert task_id == "42:batch:skeleton_inference:human_detection"


def test_make_camera_task_id_is_deterministic() -> None:
    task_id = make_camera_task_id(
        frame_number=7,
        camera_id="cam_a",
        node_kind=NODE_KIND_SKELETON_INFERENCE,
        stage="human_detection_preprocess",
    )
    assert task_id == "7:cam_a:skeleton_inference:human_detection_preprocess"


def test_normalize_parent_task_ids() -> None:
    assert normalize_parent_task_ids() == ()
    assert normalize_parent_task_ids(parent_task_id="parent") == ("parent",)
    assert normalize_parent_task_ids(parent_task_ids=["a", "b"]) == ("a", "b")
    assert normalize_parent_task_ids(
        parent_task_id="ignored",
        parent_task_ids=["a"],
    ) == ("a",)


def test_tracker_task_event_context_records_ordered_events() -> None:
    collector = TrackerTaskEventCollector()
    ctx = TrackerTaskEventContext.from_call(
        frame_number=10,
        camera_ids=["cam_0"],
        parent_task_ids=["9:cam_0:camera:human_detection"],
        event_collector=collector,
    )
    assert ctx is not None

    ctx.record_stage("human_detection_preprocess", 1_000_000, 2_000_000)
    ctx.record_stage("human_detection", 2_000_000, 5_000_000, camera_id="cam_0")

    assert len(collector.events) == 2
    batch_event, camera_event = collector.events

    assert batch_event.task_id == "10:batch:skeleton_inference:human_detection_preprocess"
    assert batch_event.parent_task_ids == ("9:cam_0:camera:human_detection",)
    assert batch_event.frame_number == 10
    assert batch_event.start_time_ns == 1_000_000
    assert batch_event.end_time_ns == 2_000_000
    assert batch_event.duration_ms == 1.0

    assert camera_event.task_id == "10:cam_0:skeleton_inference:human_detection"
    assert camera_event.camera_id == "cam_0"
    assert camera_event.duration_ms == 3.0


def test_rtmpose_batch_stage_names_match_freemocap() -> None:
    assert RTMPOSE_BATCH_STAGES == (
        "human_detection_letterbox",
        "human_detection_batch_pack",
        "human_detection_preprocess",
        "human_detection",
        "human_detection_postprocess",
        "pose_estimation_preprocess",
        "pose_estimation",
        "pose_estimation_postprocess",
    )
