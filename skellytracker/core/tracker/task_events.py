"""Task-event timing metadata for pipeline metrics integration.

Events use ``time.monotonic_ns()`` timestamps (system-wide, cross-process
comparable) and deterministic task IDs so FreeMoCap can relay them through
``pipeline_timing`` WebSocket payloads without losing parent-child links
across processes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

NODE_KIND_SKELETON_INFERENCE = "skeleton_inference"

# RTMPose predict_batch stage names (match FreeMoCap PipelineStageTimer keys).
STAGE_HUMAN_DETECTION_LETTERBOX = "human_detection_letterbox"
STAGE_HUMAN_DETECTION_BATCH_PACK = "human_detection_batch_pack"
STAGE_HUMAN_DETECTION_PREPROCESS = "human_detection_preprocess"
STAGE_HUMAN_DETECTION = "human_detection"
STAGE_HUMAN_DETECTION_POSTPROCESS = "human_detection_postprocess"
STAGE_POSE_ESTIMATION_PREPROCESS = "pose_estimation_preprocess"
STAGE_POSE_ESTIMATION = "pose_estimation"
STAGE_POSE_ESTIMATION_POSTPROCESS = "pose_estimation_postprocess"

RTMPOSE_BATCH_STAGES: tuple[str, ...] = (
    STAGE_HUMAN_DETECTION_LETTERBOX,
    STAGE_HUMAN_DETECTION_BATCH_PACK,
    STAGE_HUMAN_DETECTION_PREPROCESS,
    STAGE_HUMAN_DETECTION,
    STAGE_HUMAN_DETECTION_POSTPROCESS,
    STAGE_POSE_ESTIMATION_PREPROCESS,
    STAGE_POSE_ESTIMATION,
    STAGE_POSE_ESTIMATION_POSTPROCESS,
)


@dataclass(frozen=True)
class TrackerTaskEvent:
    task_id: str
    parent_task_ids: tuple[str, ...]
    stage: str
    node_kind: str
    camera_id: str | None
    frame_number: int | None
    start_time_ns: int
    end_time_ns: int
    duration_ms: float
    batch_index: int | None = None
    batch_size: int | None = None


@runtime_checkable
class TrackerTaskEventSink(Protocol):
    def append(self, event: TrackerTaskEvent) -> None: ...


@dataclass
class TrackerTaskEventCollector:
    """In-process sink that retains emitted task events in call order."""

    events: list[TrackerTaskEvent] = field(default_factory=list)
    max_events: int | None = None
    dropped_events: int = 0

    def append(self, event: TrackerTaskEvent) -> None:
        if self.max_events is not None and len(self.events) >= self.max_events:
            self.dropped_events += 1
            return
        self.events.append(event)


def make_batch_task_id(
    *,
    frame_number: int | None,
    node_kind: str,
    stage: str,
) -> str:
    frame = frame_number if frame_number is not None else "?"
    return f"{frame}:batch:{node_kind}:{stage}"


def make_camera_task_id(
    *,
    frame_number: int | None,
    camera_id: str,
    node_kind: str,
    stage: str,
) -> str:
    frame = frame_number if frame_number is not None else "?"
    return f"{frame}:{camera_id}:{node_kind}:{stage}"


def normalize_parent_task_ids(
    *,
    parent_task_id: str | None = None,
    parent_task_ids: list[str] | None = None,
) -> tuple[str, ...]:
    if parent_task_ids:
        return tuple(parent_task_ids)
    if parent_task_id:
        return (parent_task_id,)
    return ()


@dataclass
class TrackerTaskEventContext:
    frame_number: int | None = None
    camera_ids: list[str] | None = None
    parent_task_ids: tuple[str, ...] = ()
    event_collector: TrackerTaskEventCollector | TrackerTaskEventSink | None = None
    node_kind: str = NODE_KIND_SKELETON_INFERENCE

    @classmethod
    def from_call(
        cls,
        *,
        frame_number: int | None = None,
        camera_ids: list[str] | None = None,
        parent_task_id: str | None = None,
        parent_task_ids: list[str] | None = None,
        event_collector: TrackerTaskEventCollector | TrackerTaskEventSink | None = None,
        node_kind: str = NODE_KIND_SKELETON_INFERENCE,
    ) -> TrackerTaskEventContext | None:
        if event_collector is None:
            return None
        return cls(
            frame_number=frame_number,
            camera_ids=camera_ids,
            parent_task_ids=normalize_parent_task_ids(
                parent_task_id=parent_task_id,
                parent_task_ids=parent_task_ids,
            ),
            event_collector=event_collector,
            node_kind=node_kind,
        )

    def record_stage(
        self,
        stage: str,
        start_ns: int,
        end_ns: int,
        *,
        camera_id: str | None = None,
        batch_index: int | None = None,
        batch_size: int | None = None,
        parent_task_ids: tuple[str, ...] | None = None,
    ) -> None:
        if self.event_collector is None:
            return

        resolved_parent_task_ids = (
            parent_task_ids
            if parent_task_ids is not None
            else self.parent_task_ids
        )

        if camera_id is not None:
            task_id = make_camera_task_id(
                frame_number=self.frame_number,
                camera_id=camera_id,
                node_kind=self.node_kind,
                stage=stage,
            )
        else:
            task_id = make_batch_task_id(
                frame_number=self.frame_number,
                node_kind=self.node_kind,
                stage=stage,
            )

        duration_ms = (end_ns - start_ns) / 1e6
        self.event_collector.append(
            TrackerTaskEvent(
                task_id=task_id,
                parent_task_ids=resolved_parent_task_ids,
                stage=stage,
                node_kind=self.node_kind,
                camera_id=camera_id,
                frame_number=self.frame_number,
                start_time_ns=start_ns,
                end_time_ns=end_ns,
                duration_ms=duration_ms,
                batch_index=batch_index,
                batch_size=batch_size,
            )
        )


class StageTimer:
    """Context manager that records one stage event and optional elapsed-ms accumulator."""

    def __init__(
        self,
        ctx: TrackerTaskEventContext | None,
        stage: str,
        *,
        acc_ms: list[float] | None = None,
        camera_id: str | None = None,
        batch_index: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._ctx = ctx
        self._stage = stage
        self._acc_ms = acc_ms
        self._camera_id = camera_id
        self._batch_index = batch_index
        self._batch_size = batch_size
        self._start_ns = 0

    def __enter__(self) -> StageTimer:
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        end_ns = time.perf_counter_ns()
        duration_ms = (end_ns - self._start_ns) / 1e6
        if self._acc_ms is not None:
            self._acc_ms[0] += duration_ms
        if self._ctx is not None:
            self._ctx.record_stage(
                self._stage,
                self._start_ns,
                end_ns,
                camera_id=self._camera_id,
                batch_index=self._batch_index,
                batch_size=self._batch_size,
            )
