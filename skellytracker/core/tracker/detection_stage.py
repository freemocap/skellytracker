from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import NDArray
import numpy as np

from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.data_primitives import BoundingBox, Keypoints
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import (
    KeypointDetector,
    ObjectDetector,
    build_keypoint_detector,
    build_object_detector,
)
from skellytracker.core.observation import StageObservation
from skellytracker.core.sessions.session import Session
from skellytracker.core.tracker.tracker_state import KeypointSmoothingState, StageState


@dataclass
class DetectionStage:
    """Compositional unit of the detection pipeline.

    Binds one optional ObjectDetector with one or more KeypointDetectors.
    Child stages receive the parent's crop and keypoints as context and run
    their own detection subtree, enabling hierarchical top-down pipelines
    (e.g., body stage → face child stage).
    """

    name: str
    keypoint_detectors: list[KeypointDetector]
    object_detector: ObjectDetector | None = None
    children: list[DetectionStage] = field(default_factory=list)

    def run(
        self,
        image: NDArray[np.uint8],
        state: StageState,
        parent_keypoints: Keypoints | None = None,
        context: DetectionContext | None = None,
    ) -> tuple[StageObservation, StageState]:
        """Run this stage and all child stages.

        Args:
            image: Full or parent-cropped image.
            state: Current temporal state for this stage.
            parent_keypoints: Keypoints from the parent stage, available for
                              computing crop regions (e.g., wrist → hand crop).

        Returns:
            (StageObservation, updated StageState)
        """
        # 1. Object detection
        if self.object_detector is not None:
            bboxes = self.object_detector.detect(image, context)
        else:
            h, w = image.shape[:2]
            bboxes = [BoundingBox.full_image(h, w)]

        bbox = bboxes[0] if bboxes else None  # Change this to support multiple people/objects

        # 2. Keypoint detection — merge multiple detectors into one Keypoints
        all_keypoints: list[Keypoints] = []
        updated_kp_states: list = []
        for i, detector in enumerate(self.keypoint_detectors):
            kp_state = (
                state.keypoint_states[i]
                if i < len(state.keypoint_states)
                else KeypointSmoothingState()
            )
            crop = bbox.to_crop(image) if bbox is not None else image
            kpts = detector.detect(crop, bbox, context)
            all_keypoints.append(kpts)
            updated_kp_states.append(kp_state)

        merged = Keypoints.concatenate(all_keypoints) if all_keypoints else None

        # 3. Child stages
        child_observations: dict[str, StageObservation] = {}
        updated_child_states: dict[str, StageState] = {}
        for child in self.children:
            child_state = state.child_states.get(child.name, StageState())
            crop = bbox.to_crop(image) if bbox is not None else image
            child_obs, child_state = child.run(crop, child_state, parent_keypoints=merged, context=context)
            child_observations[child.name] = child_obs
            updated_child_states[child.name] = child_state

        obs = StageObservation(
            name=self.name,
            bounding_boxes=bboxes,
            keypoints=merged,
            children=child_observations,
        )
        updated_state = StageState(
            bbox_state=state.bbox_state,
            keypoint_states=updated_kp_states,
            child_states=updated_child_states,
        )
        return obs, updated_state

    def close(self) -> None:
        """Release resources owned by all detectors in this stage and its children."""
        if self.object_detector is not None:
            self.object_detector.close()
        for detector in self.keypoint_detectors:
            detector.close()
        for child in self.children:
            child.close()

    @classmethod
    def create(
        cls,
        config: DetectionStageConfig,
        sessions: dict[str, Session],
    ) -> DetectionStage:
        """Build a DetectionStage and its full subtree from config."""
        object_detector = (
            build_object_detector(config.object_detector, sessions)
            if config.object_detector is not None
            else None
        )
        keypoint_detectors = [
            build_keypoint_detector(kp_cfg, sessions)
            for kp_cfg in config.keypoint_detectors
        ]
        children = [
            DetectionStage.create(child_cfg, sessions)
            for child_cfg in config.children
        ]
        return cls(
            name=config.name,
            object_detector=object_detector,
            keypoint_detectors=keypoint_detectors,
            children=children,
        )
