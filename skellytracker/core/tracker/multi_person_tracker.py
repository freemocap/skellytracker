from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import NDArray
import numpy as np

from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.data_primitives.multi_person_observation import MultiPersonObservation
from skellytracker.core.data_primitives.observation import Observation, StageObservation
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.sessions.session import Session
from skellytracker.core.temporal_processing.multi_person_config import MultiPersonTrackingConfig
from skellytracker.core.temporal_processing.track_association import associate
from skellytracker.core.tracker.detection_stage import DetectionStage, RawStageDetection
from skellytracker.core.tracker.person_track import PersonTrackState
from skellytracker.core.tracker.tracker_state import StageState


@dataclass
class MultiPersonTracker:
    """Top-level orchestrator for multi-person, single-camera temporal tracking.

    Runs the root stage's object detector once per frame to propose candidate
    person boxes, runs each candidate's keypoint detection independent of
    identity, associates candidates to existing PersonTrackStates (IoU +
    keypoint-distance cost, Hungarian assignment — see track_association.py),
    then finalizes each match using that track's own accumulated temporal
    state (bbox EMA, keypoint smoothing, reset policy) — the same per-subject
    machinery Tracker/DetectionStage use for single-person tracking, just
    keyed per track instead of per frame.

    Unlike Tracker, the root DetectionStage's own bbox-reuse policy is not
    used: the object detector runs on every frame so every person present can
    be discovered, matched, or spawned as a new track.
    """

    stages: list[DetectionStage]
    multi_person_config: MultiPersonTrackingConfig = field(default_factory=MultiPersonTrackingConfig)
    sessions: dict[str, Session] = field(default_factory=dict)
    _next_track_id: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        # Only a single top-level stage tree is supported: candidate proposals
        # come from this stage's own object detector, and its children (if
        # any, e.g. hand/face sub-stages) are run per matched track. Multiple
        # independent top-level stages — supported by Tracker — aren't wired
        # into association here.
        if len(self.stages) != 1:
            raise ValueError(
                "MultiPersonTracker supports exactly one top-level stage "
                f"(with children for sub-stages), got {len(self.stages)}"
            )
        if self.stages[0].object_detector is None:
            raise ValueError(
                "MultiPersonTracker's root stage must have an object_detector "
                "to propose per-frame candidate person boxes"
            )

    def process_image(
        self,
        image: NDArray[np.uint8],
        frame_number: int,
        tracks: dict[int, PersonTrackState],
        timestamp_ms: int | None = None,
    ) -> tuple[MultiPersonObservation, dict[int, PersonTrackState]]:
        h, w = image.shape[:2]
        context = DetectionContext(frame_number=frame_number, timestamp_ms=timestamp_ms)
        root = self.stages[0]

        # 1. Propose candidate person boxes for the whole frame.
        candidate_bboxes = root.object_detector.detect(image, context)

        # 2. Run keypoint detection per candidate, independent of identity —
        # every downstream stage (root + children) computes raw detections,
        # keyed by stage name, so association can use any stage's keypoints.
        candidate_raw: list[dict[str, RawStageDetection]] = [
            self._detect_raw_tree(bbox, image, context) for bbox in candidate_bboxes
        ]
        candidate_keypoints: list[Keypoints | None] = [
            Keypoints.concatenate(raw[root.name].keypoints) if raw[root.name].keypoints else None
            for raw in candidate_raw
        ]

        # 3. Associate this frame's candidates against existing tracks.
        track_ids = list(tracks.keys())
        track_bboxes: list[BoundingBox | None] = [tracks[tid].last_bbox for tid in track_ids]
        track_keypoints: list[Keypoints | None] = [tracks[tid].last_keypoints for tid in track_ids]
        result = associate(
            track_bboxes, track_keypoints, candidate_bboxes, candidate_keypoints, self.multi_person_config
        )

        # 4. Matched: finalize using the track's own accumulated temporal state.
        for track_idx, det_idx in result.matches:
            track_id = track_ids[track_idx]
            track = tracks[track_id]
            self._finalize_match(track, candidate_raw[det_idx], context)

        # 5. Unmatched detections: spawn new tracks.
        for det_idx in result.unmatched_detections:
            track_id = self._next_track_id
            self._next_track_id += 1
            track = PersonTrackState(track_id=track_id)
            self._finalize_match(track, candidate_raw[det_idx], context)
            tracks[track_id] = track

        # 6. Unmatched tracks: age out.
        for track_idx in result.unmatched_tracks:
            track_id = track_ids[track_idx]
            track = tracks[track_id]
            track.time_since_update += 1
            track.age += 1

        for track_id in [tid for tid, t in tracks.items() if t.time_since_update > self.multi_person_config.max_age]:
            del tracks[track_id]

        # 7. Emit confirmed tracks that matched this frame.
        people = {}
        for track_id, track in tracks.items():
            if track.time_since_update == 0 and track.confirmed(self.multi_person_config.min_hits):
                people[track_id] = self._observation_from_track(track, frame_number, (h, w))

        observation = MultiPersonObservation(frame_number=frame_number, image_size=(h, w), people=people)
        return observation, tracks

    def _detect_raw_tree(
        self,
        bbox: BoundingBox,
        image: NDArray[np.uint8],
        context: DetectionContext,
    ) -> dict[str, RawStageDetection]:
        """Run detect_raw_at_bbox recursively over the stage tree, rooted at `bbox`."""
        raw_by_stage: dict[str, RawStageDetection] = {}

        def _recurse(stage: DetectionStage, crop_bbox: BoundingBox, source_image: NDArray[np.uint8]) -> None:
            raw = stage.detect_raw_at_bbox(source_image, crop_bbox, context)
            raw_by_stage[stage.name] = raw
            for child in stage.children:
                h, w = raw.crop.shape[:2]
                _recurse(child, BoundingBox.full_image(h, w), raw.crop)

        _recurse(self.stages[0], bbox, image)
        return raw_by_stage

    def _finalize_match(
        self,
        track: PersonTrackState,
        raw_by_stage: dict[str, RawStageDetection],
        context: DetectionContext,
    ) -> None:
        """Apply per-track temporal smoothing across the whole stage tree in place."""

        def _recurse(stage: DetectionStage) -> None:
            state = track.stage_states.get(stage.name, StageState())
            raw = raw_by_stage[stage.name]
            obs, updated_state = stage.finalize_track(raw, state, context)
            track.stage_states[stage.name] = updated_state
            if stage.name == self.stages[0].name:
                track.last_bbox = obs.bounding_boxes[0] if obs.bounding_boxes else raw.bbox
                track.last_keypoints = obs.keypoints
            for child in stage.children:
                _recurse(child)

        _recurse(self.stages[0])
        track.hits += 1
        track.time_since_update = 0
        track.age += 1

    def _observation_from_track(
        self,
        track: PersonTrackState,
        frame_number: int,
        image_size: tuple[int, int],
    ) -> Observation:
        def _build(stage: DetectionStage) -> StageObservation:
            state = track.stage_states.get(stage.name, StageState())
            children = {child.name: _build(child) for child in stage.children}
            bbox = state.bbox_state.smooth_bbox
            return StageObservation(
                name=stage.name,
                bounding_boxes=[bbox] if bbox is not None else [],
                keypoints=state.last_keypoints,
                children=children,
                detector_ran=True,
            )

        stages = {self.stages[0].name: _build(self.stages[0])}
        return Observation(frame_number=frame_number, image_size=image_size, stages=stages)

    def close(self) -> None:
        for stage in self.stages:
            stage.close()
        for session in self.sessions.values():
            session.close()

    def reset_temporal_state(self) -> None:
        for stage in self.stages:
            stage.reset_temporal_state()

    @classmethod
    def create(
        cls,
        config: TrackerConfig,
        sessions: dict[str, Session],
        multi_person_config: MultiPersonTrackingConfig | None = None,
    ) -> MultiPersonTracker:
        stages = [DetectionStage.create(stage_cfg, sessions) for stage_cfg in config.stages]
        return cls(
            stages=stages,
            multi_person_config=multi_person_config or MultiPersonTrackingConfig(),
            sessions=sessions,
        )
