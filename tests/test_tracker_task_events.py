"""Unit tests for tracker task-event helpers."""
from skellytracker.core.tracker.task_events import (
    NODE_KIND_SKELETON_INFERENCE,
    RTMPOSE_BATCH_STAGES,
    StageTimer,
    TrackerTaskEvent,
    TrackerTaskEventCollector,
    TrackerTaskEventContext,
    make_batch_task_id,
    make_camera_task_id,
    normalize_parent_task_ids,
)


# ── Task ID construction ──────────────────────────────────────────────────────

def test_make_batch_task_id_is_deterministic() -> None:
    task_id = make_batch_task_id(
        frame_number=42,
        node_kind=NODE_KIND_SKELETON_INFERENCE,
        stage="human_detection",
    )
    assert task_id == "42:batch:skeleton_inference:human_detection"


def test_make_batch_task_id_with_none_frame_number() -> None:
    task_id = make_batch_task_id(
        frame_number=None,
        node_kind=NODE_KIND_SKELETON_INFERENCE,
        stage="human_detection",
    )
    assert task_id == "?:batch:skeleton_inference:human_detection"


def test_make_camera_task_id_is_deterministic() -> None:
    task_id = make_camera_task_id(
        frame_number=7,
        camera_id="cam_a",
        node_kind=NODE_KIND_SKELETON_INFERENCE,
        stage="human_detection_preprocess",
    )
    assert task_id == "7:cam_a:skeleton_inference:human_detection_preprocess"


def test_make_camera_task_id_with_none_frame_number() -> None:
    task_id = make_camera_task_id(
        frame_number=None,
        camera_id="cam_a",
        node_kind=NODE_KIND_SKELETON_INFERENCE,
        stage="human_detection_preprocess",
    )
    assert task_id == "?:cam_a:skeleton_inference:human_detection_preprocess"


# ── Parent task-id normalisation ──────────────────────────────────────────────

def test_normalize_parent_task_ids() -> None:
    assert normalize_parent_task_ids() == ()
    assert normalize_parent_task_ids(parent_task_id="parent") == ("parent",)
    assert normalize_parent_task_ids(parent_task_ids=["a", "b"]) == ("a", "b")
    assert normalize_parent_task_ids(
        parent_task_id="ignored",
        parent_task_ids=["a"],
    ) == ("a",)


def test_normalize_both_explicitly_none() -> None:
    """Both params explicitly None returns empty tuple (same as no-args)."""
    assert normalize_parent_task_ids(
        parent_task_id=None,
        parent_task_ids=None,
    ) == ()


# ── TrackerTaskEventCollector ─────────────────────────────────────────────────

def test_collector_with_max_events_drops_overflow() -> None:
    collector = TrackerTaskEventCollector(max_events=3)

    for i in range(5):
        collector.append(TrackerTaskEvent(
            task_id=f"{i}:batch:skeleton_inference:stage",
            parent_task_ids=(),
            stage="stage",
            node_kind=NODE_KIND_SKELETON_INFERENCE,
            camera_id=None,
            frame_number=i,
            start_time_ns=i * 100,
            end_time_ns=(i + 1) * 100,
            duration_ms=0.1,
        ))
    assert len(collector.events) == 3
    assert collector.dropped_events == 2
    # First three events are retained.
    assert [e.frame_number for e in collector.events] == [0, 1, 2]


# ── TrackerTaskEventContext ───────────────────────────────────────────────────

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


def test_context_from_call_with_no_collector_returns_none() -> None:
    ctx = TrackerTaskEventContext.from_call(
        frame_number=10,
        camera_ids=["cam_0"],
        parent_task_ids=["parent"],
        event_collector=None,
    )
    assert ctx is None


def test_record_stage_with_explicit_parent_task_ids() -> None:
    collector = TrackerTaskEventCollector()
    ctx = TrackerTaskEventContext.from_call(
        frame_number=1,
        camera_ids=["cam_0"],
        parent_task_ids=["default_parent"],
        event_collector=collector,
    )
    assert ctx is not None

    ctx.record_stage(
        "custom_stage", 100, 200,
        parent_task_ids=("override_parent",),
    )
    assert len(collector.events) == 1
    assert collector.events[0].parent_task_ids == ("override_parent",)


def test_record_stage_with_batch_metadata() -> None:
    collector = TrackerTaskEventCollector()
    ctx = TrackerTaskEventContext.from_call(
        frame_number=1,
        camera_ids=["cam_0"],
        parent_task_ids=["parent"],
        event_collector=collector,
    )
    assert ctx is not None

    ctx.record_stage("batch_stage", 100, 200, batch_index=0, batch_size=4)
    assert len(collector.events) == 1
    event = collector.events[0]
    assert event.batch_index == 0
    assert event.batch_size == 4


# ── StageTimer context manager ────────────────────────────────────────────────

def test_stage_timer_context_manager_records_event() -> None:
    collector = TrackerTaskEventCollector()
    ctx = TrackerTaskEventContext.from_call(
        frame_number=5,
        camera_ids=["cam_0"],
        parent_task_ids=["parent"],
        event_collector=collector,
    )
    assert ctx is not None

    with StageTimer(ctx, "human_detection", camera_id="cam_0", batch_index=0, batch_size=2):
        pass  # simulated work

    assert len(collector.events) == 1
    event = collector.events[0]
    assert event.stage == "human_detection"
    assert event.camera_id == "cam_0"
    assert event.batch_index == 0
    assert event.batch_size == 2
    assert event.frame_number == 5
    assert event.start_time_ns > 0
    assert event.end_time_ns >= event.start_time_ns
    assert event.duration_ms >= 0.0


def test_stage_timer_is_noop_when_ctx_is_none() -> None:
    """StageTimer with ctx=None must not raise and must not record anything."""
    with StageTimer(None, "human_detection"):
        pass  # should be safe


def test_stage_timer_accumulates_into_acc_ms() -> None:
    collector = TrackerTaskEventCollector()
    ctx = TrackerTaskEventContext.from_call(
        frame_number=5,
        camera_ids=["cam_0"],
        parent_task_ids=["parent"],
        event_collector=collector,
    )
    assert ctx is not None

    acc = [0.0]
    with StageTimer(ctx, "human_detection", acc_ms=acc):
        pass

    assert acc[0] > 0.0  # duration was added
    assert len(collector.events) == 1


# ── RTMPOSE batch stages ──────────────────────────────────────────────────────

def test_rtmpose_batch_stages_are_nonempty_and_ordered() -> None:
    assert len(RTMPOSE_BATCH_STAGES) > 0
    # The canonical stage ordering must be preserved.
    assert RTMPOSE_BATCH_STAGES.index("human_detection_letterbox") == 0
    assert RTMPOSE_BATCH_STAGES.index("pose_estimation_postprocess") == len(RTMPOSE_BATCH_STAGES) - 1


def test_rtmpose_stages_map_to_valid_attr_names() -> None:
    """Each stage must correspond to a plausible ``last_<stage>_ms`` attribute."""
    for stage in RTMPOSE_BATCH_STAGES:
        attr = f"last_{stage}_ms"
        assert attr.replace(" ", "_") == attr, (
            f"Stage name '{stage}' produces invalid attribute name '{attr}'"
        )
