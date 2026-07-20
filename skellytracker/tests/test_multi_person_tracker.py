from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.config.detector_configs import KeypointDetectorConfig, ObjectDetectorConfig
from skellytracker.core.config.session_config import SessionConfig
from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import KeypointDetector, ObjectDetector
from skellytracker.core.sessions.session import Session
from skellytracker.core.temporal_processing.multi_person_config import MultiPersonTrackingConfig
from skellytracker.core.tracker.detection_stage import DetectionStage
from skellytracker.core.tracker.multi_person_tracker import MultiPersonTracker
from skellytracker.core.tracker.person_track import PersonTrackState


# ---------------------------------------------------------------------------
# Test doubles — no ONNX/mediapipe required, fully scripted per frame_number.
# ---------------------------------------------------------------------------

class _FakeObjectDetectorConfig(ObjectDetectorConfig):
    detector_type: str = "fake_object"
    session_backend: str = "fake"


class _FakeKeypointDetectorConfig(KeypointDetectorConfig):
    detector_type: str = "fake_keypoint"
    session_backend: str = "fake"


@dataclass
class _FakeSession(Session):
    @classmethod
    def create(cls, config: SessionConfig) -> "_FakeSession":
        return cls()

    def close(self) -> None:
        pass


@dataclass
class ScriptedObjectDetector(ObjectDetector):
    """Returns a scripted list of boxes for each frame_number."""

    script: dict[int, list[BoundingBox]] = field(default_factory=dict)

    def detect(
        self, image: NDArray[np.uint8], context: DetectionContext | None = None
    ) -> list[BoundingBox]:
        frame = context.frame_number if context is not None else 0
        return list(self.script.get(frame, []))

    def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], Any]:
        raise NotImplementedError

    def postprocess(self, raw: Any, metadata: Any) -> list[BoundingBox]:
        raise NotImplementedError

    @classmethod
    def create(cls, config: ObjectDetectorConfig, session: Session) -> "ScriptedObjectDetector":
        raise NotImplementedError


@dataclass
class CenterPointKeypointDetector(KeypointDetector):
    """Deterministic single-point detector: always the center of the crop."""

    def detect(
        self, image: NDArray[np.uint8], context: DetectionContext | None = None
    ) -> Keypoints:
        h, w = image.shape[:2]
        return Keypoints(
            names=("center",),
            xyz=np.array([[w / 2.0, h / 2.0, 0.0]]),
            visibility=np.array([1.0]),
        )

    def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], Any]:
        raise NotImplementedError

    def postprocess(self, raw: Any, metadata: Any) -> Keypoints:
        raise NotImplementedError

    @classmethod
    def create(cls, config: KeypointDetectorConfig, session: Session) -> "CenterPointKeypointDetector":
        raise NotImplementedError


def _box(cx: float, cy: float, size: float = 50.0) -> BoundingBox:
    return BoundingBox(x1=cx - size / 2, y1=cy - size / 2, x2=cx + size / 2, y2=cy + size / 2)


def _make_tracker(script: dict[int, list[BoundingBox]], **config_kwargs) -> MultiPersonTracker:
    session = _FakeSession()
    stage = DetectionStage(
        name="body",
        object_detector=ScriptedObjectDetector(
            config=_FakeObjectDetectorConfig(), session=session, script=script
        ),
        keypoint_detectors=[CenterPointKeypointDetector(config=_FakeKeypointDetectorConfig(), session=session)],
    )
    config = MultiPersonTrackingConfig(min_hits=1, max_age=2, max_match_cost=0.6, **config_kwargs)
    return MultiPersonTracker(stages=[stage], multi_person_config=config)


_IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiPersonTracker:
    def test_two_people_get_stable_distinct_track_ids_across_frames(self):
        # Two people, non-overlapping, drifting slowly — 6 frames.
        script = {
            f: [_box(50 + f * 5, 50 + f * 5), _box(400 + f * 5, 300 + f * 5)]
            for f in range(6)
        }
        tracker = _make_tracker(script)
        tracks: dict[int, PersonTrackState] = {}

        seen_ids_per_frame = []
        for f in range(6):
            obs, tracks = tracker.process_image(_IMAGE, f, tracks)
            seen_ids_per_frame.append(set(obs.people.keys()))

        # Same two track IDs present every frame from the first frame onward
        # (min_hits=1), and no extra tracks spawned for smoothly moving people.
        assert all(ids == seen_ids_per_frame[0] for ids in seen_ids_per_frame)
        assert len(seen_ids_per_frame[0]) == 2

    def test_track_dropped_after_max_age_then_reappearance_gets_new_id(self):
        # Person present frames 0-2, then gone for > max_age(=2) frames, then
        # reappears at a totally different location.
        script = {
            0: [_box(50, 50)],
            1: [_box(55, 55)],
            2: [_box(60, 60)],
            # frames 3, 4, 5: no detections (person absent)
            6: [_box(500, 400)],
        }
        tracker = _make_tracker(script)
        tracks: dict[int, PersonTrackState] = {}

        first_id = None
        for f in range(3):
            obs, tracks = tracker.process_image(_IMAGE, f, tracks)
            first_id = next(iter(obs.people))

        for f in range(3, 6):
            obs, tracks = tracker.process_image(_IMAGE, f, tracks)
            assert obs.people == {}

        # Track should have aged out (time_since_update > max_age) by frame 6.
        assert first_id not in tracks

        obs, tracks = tracker.process_image(_IMAGE, 6, tracks)
        assert len(obs.people) == 1
        new_id = next(iter(obs.people))
        assert new_id != first_id

    def test_crossing_detections_prefer_nearest_iou_match(self):
        # Two people approach but stay non-overlapping (max IoU association is
        # unambiguous each frame); confirms association doesn't scramble IDs
        # just because both are moving toward each other.
        script = {
            0: [_box(50, 50), _box(450, 50)],
            1: [_box(60, 50), _box(440, 50)],
            2: [_box(70, 50), _box(430, 50)],
        }
        tracker = _make_tracker(script)
        tracks: dict[int, PersonTrackState] = {}

        id_positions: list[dict[int, float]] = []
        for f in range(3):
            obs, tracks = tracker.process_image(_IMAGE, f, tracks)
            positions = {}
            for track_id, person_obs in obs.people.items():
                bbox = person_obs.stages["body"].bounding_boxes[0]
                positions[track_id] = bbox.center[0]
            id_positions.append(positions)

        assert len(id_positions[0]) == 2
        left_id = min(id_positions[0], key=lambda tid: id_positions[0][tid])
        right_id = max(id_positions[0], key=lambda tid: id_positions[0][tid])

        # The left-starting track's x position should keep increasing (moving
        # right), and the right-starting track's should keep decreasing.
        left_xs = [frame[left_id] for frame in id_positions]
        right_xs = [frame[right_id] for frame in id_positions]
        assert left_xs == sorted(left_xs)
        assert right_xs == sorted(right_xs, reverse=True)
