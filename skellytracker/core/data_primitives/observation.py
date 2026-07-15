from __future__ import annotations

from dataclasses import dataclass, field

from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.data_primitives.keypoints import Keypoints


@dataclass
class StageObservation:
    """Detection output from a single DetectionStage."""

    name: str
    bounding_boxes: list[BoundingBox] = field(default_factory=list)
    keypoints: Keypoints | None = None
    children: dict[str, StageObservation] = field(default_factory=dict)


@dataclass
class Observation:
    """Per-frame output of a Tracker.

    The contract between the tracking pipeline and all downstream consumers
    (annotators, data stores, freemocap triangulation). Callers should not
    need to know which detectors produced the data.
    """

    frame_number: int
    image_size: tuple[int, int]              # (height, width) in pixels
    stages: dict[str, StageObservation] = field(default_factory=dict)
    timestamp: float | None = None

    def to_keypoints(self) -> Keypoints:
        """Merge all stage keypoints into a single flat Keypoints instance.

        Point names are prefixed by stage name: "<stage>.<point>".
        Traverses the full stage tree (including children) in config-determined
        order. This is the form passed to freemocap for triangulation.
        """
        clouds: list[Keypoints] = []
        self._collect_keypoints(self.stages, clouds)
        if not clouds:
            return Keypoints.empty(())
        return Keypoints.concatenate(clouds)

    def _collect_keypoints(
        self,
        stages: dict[str, StageObservation],
        out: list[Keypoints],
    ) -> None:
        for stage in stages.values():
            if stage.keypoints is not None:
                prefixed = Keypoints(
                    names=tuple(f"{stage.name}.{n}" for n in stage.keypoints.names),
                    xyz=stage.keypoints.xyz,
                    visibility=stage.keypoints.visibility,
                )
                out.append(prefixed)
            self._collect_keypoints(stage.children, out)
