from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import NDArray
import numpy as np

from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.detectors.detection_context import DetectionContext
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

    def close(self) -> None:
        """Release all detector and session resources."""
        for stage in self.stages:
            stage.close()
        for session in self.sessions.values():
            session.close()

    @classmethod
    def create(cls, config: TrackerConfig, sessions: dict[str, Session]) -> Tracker:
        """Build a Tracker from config and pre-created sessions."""
        stages = [DetectionStage.create(stage_cfg, sessions) for stage_cfg in config.stages]
        return cls(stages=stages, sessions=sessions)
