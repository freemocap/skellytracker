from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import NDArray
import numpy as np

from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.processing_timer import ProcessingTimer
from skellytracker.core.tracker.detection_stage import DetectionStage
from skellytracker.core.observation import Observation
from skellytracker.core.sessions.session import Session
from skellytracker.core.tracker.tracker_state import StageState, TrackerState


@dataclass
class Tracker:
    """Top-level pipeline orchestrator.

    Takes images, runs all DetectionStages in order, and returns a structured
    Observation plus updated TrackerState. Stateless between calls — all
    temporal data lives in TrackerState.
    """

    stages: list[DetectionStage]
    sessions: dict[str, Session] = field(default_factory=dict)

    def process_image(
        self,
        image: NDArray[np.uint8],
        frame_number: int,
        state: TrackerState,
        timestamp_ms: int | None = None,
    ) -> tuple[Observation, TrackerState]:
        """Run all stages on a frame and return the merged Observation.

        Args:
            image: BGR image array (H, W, 3).
            frame_number: Frame index; recorded in the Observation.
            state: Current TrackerState; returned updated.
            timestamp_ms: Monotonically increasing wall-clock time in ms.
                          Required by detectors in VIDEO mode. When None,
                          detectors derive their own timestamp.

        Returns:
            (Observation for this frame, updated TrackerState)
        """
        h, w = image.shape[:2]
        context = DetectionContext(frame_number=frame_number, timestamp_ms=timestamp_ms)
        stage_observations = {}
        updated_stage_states: dict[str, StageState] = {}

        for stage in self.stages:
            stage_state = state.stage_states.get(stage.name, StageState())
            stage_obs, stage_state = stage.run(image, stage_state, context=context)
            stage_observations[stage.name] = stage_obs
            updated_stage_states[stage.name] = stage_state

        observation = Observation(
            frame_number=frame_number,
            image_size=(h, w),
            stages=stage_observations,
        )
        updated_state = TrackerState(stage_states=updated_stage_states)
        return observation, updated_state

    def process_batch(
        self,
        images: dict[str, NDArray[np.uint8]],
        frame_number: int,
        states: dict[str, TrackerState],
        timestamp_ms: int | None = None,
        timings: ProcessingTimer | None = None,
    ) -> tuple[dict[str, Observation], dict[str, TrackerState]]:
        """Run all stages on N cameras simultaneously.

        For ONNX-backed detectors, cameras are processed in a single batched
        ORT call. For non-ONNX detectors a thread pool is used. Child stage
        hierarchies are preserved.

        Args:
            images:
                Mapping from camera ID to BGR image array (H, W, 3).
                All images must have the same spatial dimensions.
            frame_number:
                Frame index; recorded in every per-camera Observation.
            states:
                Mapping from camera ID to current TrackerState. Keys may be a
                subset of images.keys() — missing cameras start with empty state.
            timestamp_ms:
                Monotonically increasing wall-clock time in ms. Passed to all
                detectors unchanged (used by MediaPipe VIDEO mode).

        Returns:
            (per-camera Observation dict, per-camera updated TrackerState dict)
        """
        if not images:
            return {}, {}

        first_image = next(iter(images.values()))
        h, w = first_image.shape[:2]
        context = DetectionContext(frame_number=frame_number, timestamp_ms=timestamp_ms, timings=timings)
        cam_ids = list(images.keys())

        # per-camera accumulators
        per_cam_stage_obs: dict[str, dict] = {cam_id: {} for cam_id in cam_ids}
        per_cam_stage_states: dict[str, dict] = {cam_id: {} for cam_id in cam_ids}

        for stage in self.stages:
            stage_states_for_batch = {
                cam_id: states.get(cam_id, TrackerState()).stage_states.get(stage.name, StageState())
                for cam_id in cam_ids
            }
            stage_obs_batch, stage_states_batch = stage.run_batch(images, stage_states_for_batch, context)
            for cam_id in cam_ids:
                per_cam_stage_obs[cam_id][stage.name] = stage_obs_batch[cam_id]
                per_cam_stage_states[cam_id][stage.name] = stage_states_batch[cam_id]

        observations = {
            cam_id: Observation(
                frame_number=frame_number,
                image_size=(h, w),
                stages=per_cam_stage_obs[cam_id],
            )
            for cam_id in cam_ids
        }
        updated_states = {
            cam_id: TrackerState(stage_states=per_cam_stage_states[cam_id])
            for cam_id in cam_ids
        }
        return observations, updated_states

    def close(self) -> None:
        """Release all detector and session resources."""
        for stage in self.stages:
            stage.close()
        for session in self.sessions.values():
            session.close()

    def reset_temporal_state(self) -> None:
        """Reset internal temporal state on all detectors across all stages.

        Call this between independent videos when reusing a tracker across
        multiple files — stateful backends (e.g. MediaPipe VIDEO mode) will
        start each video from a clean state rather than trying to track
        across the file boundary.
        """
        for stage in self.stages:
            stage.reset_temporal_state()

    @classmethod
    def create(cls, config: TrackerConfig, sessions: dict[str, Session]) -> Tracker:
        """Build a Tracker from config and pre-created sessions."""
        stages = [DetectionStage.create(stage_cfg, sessions) for stage_cfg in config.stages]
        return cls(stages=stages, sessions=sessions)
