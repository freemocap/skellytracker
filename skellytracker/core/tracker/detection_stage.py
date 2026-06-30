from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import NDArray
import numpy as np

from skellytracker.core.data_primitives import BoundingBox, Keypoints
from skellytracker.core.detectors.detector_base_classes import KeypointDetector, ObjectDetector
from skellytracker.core.observation import StageObservation
from skellytracker.core.tracker.tracker_state import StageState


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
            bboxes = self.object_detector.detect(image)
        else:
            h, w = image.shape[:2]
            bboxes = [BoundingBox.full_image(h, w)]

        bbox = bboxes[0] if bboxes else None

        # 2. Keypoint detection — merge multiple detectors into one Keypoints
        all_keypoints: list[Keypoints] = []
        updated_kp_states: list = []
        for detector, kp_state in zip(self.keypoint_detectors, state.keypoint_states):
            crop = bbox.to_crop(image) if bbox is not None else image
            kpts = detector.detect(crop, bbox)
            all_keypoints.append(kpts)
            updated_kp_states.append(kp_state)

        merged = Keypoints.concatenate(all_keypoints) if all_keypoints else None

        # 3. Child stages
        child_observations: dict[str, StageObservation] = {}
        updated_child_states: dict[str, StageState] = {}
        for child in self.children:
            child_state = state.child_states.get(child.name, StageState())
            crop = bbox.to_crop(image) if bbox is not None else image
            child_obs, child_state = child.run(crop, child_state, parent_keypoints=merged)
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
