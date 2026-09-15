from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from skellytracker.core.config.detector_configs import KeypointDetectorConfig
from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import KeypointDetector
from skellytracker.core.detectors.object_detectors.keypoint_bbox import (
    KeypointBoundingBoxDetector,
    KeypointBoundingBoxDetectorConfig,
)
from skellytracker.core.sessions.cpu_session import CpuSession
from skellytracker.core.sessions.session import Session
from skellytracker.core.tracker.detection_stage import DetectionStage
from skellytracker.core.tracker.tracker_state import StageState


class _FakeKeypointDetectorConfig(KeypointDetectorConfig):
    detector_type: str = "fake_keypoint"
    session_backend: str = "fake"


@dataclass
class _WristKeypointDetector(KeypointDetector):
    """Parent-stage stand-in: always reports a fixed right_wrist point."""

    def detect(
        self, image: NDArray[np.uint8], context: DetectionContext | None = None
    ) -> Keypoints:
        return Keypoints(
            names=("right_wrist",),
            xyz=np.array([[200.0, 150.0, 0.0]]),
            visibility=np.array([1.0]),
        )

    def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], Any]:
        raise NotImplementedError

    def postprocess(self, raw: Any, metadata: Any) -> Keypoints:
        raise NotImplementedError

    @classmethod
    def create(cls, config: KeypointDetectorConfig, session: Session) -> "_WristKeypointDetector":
        return cls(config=config, session=session)


@dataclass
class _RecordingKeypointDetector(KeypointDetector):
    """Child-stage spy: records the shape of every crop it's given."""

    seen_crop_shapes: list[tuple[int, int]] = field(default_factory=list)

    def detect(
        self, image: NDArray[np.uint8], context: DetectionContext | None = None
    ) -> Keypoints:
        self.seen_crop_shapes.append(image.shape[:2])
        return Keypoints(names=(), xyz=np.zeros((0, 3)), visibility=np.zeros(0))

    def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], Any]:
        raise NotImplementedError

    def postprocess(self, raw: Any, metadata: Any) -> Keypoints:
        raise NotImplementedError

    @classmethod
    def create(cls, config: KeypointDetectorConfig, session: Session) -> "_RecordingKeypointDetector":
        # run_batch's non-ONNX path builds one detector instance per camera via
        # create(), so each gets its own seen_crop_shapes list — batch-path
        # tests assert on the returned bounding boxes instead of this list.
        return cls(config=config, session=session)


class _FixedPointKeypointDetectorConfig(KeypointDetectorConfig):
    detector_type: str = "fixed_point"
    session_backend: str = "fake"
    point_name: str = "right_wrist"
    xy: tuple[float, float] = (200.0, 150.0)


@dataclass
class _FixedPointKeypointDetector(KeypointDetector):
    """Parent-stage stand-in: always reports a fixed named point.

    The point comes off `config` (not an instance field) so it survives
    run_batch's per-camera `type(detector).create(detector.config, ...)`
    reconstruction.
    """

    def detect(
        self, image: NDArray[np.uint8], context: DetectionContext | None = None
    ) -> Keypoints:
        cfg = self.config
        assert isinstance(cfg, _FixedPointKeypointDetectorConfig)
        return Keypoints(
            names=(cfg.point_name,),
            xyz=np.array([[cfg.xy[0], cfg.xy[1], 0.0]]),
            visibility=np.array([1.0]),
        )

    def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], Any]:
        raise NotImplementedError

    def postprocess(self, raw: Any, metadata: Any) -> Keypoints:
        raise NotImplementedError

    @classmethod
    def create(cls, config: KeypointDetectorConfig, session: Session) -> "_FixedPointKeypointDetector":
        return cls(config=config, session=session)


@pytest.fixture
def cpu_session() -> CpuSession:
    return CpuSession()


@pytest.fixture
def image() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _build_hierarchy(cpu_session: CpuSession) -> tuple[DetectionStage, _RecordingKeypointDetector]:
    recorder = _RecordingKeypointDetector(config=_FakeKeypointDetectorConfig(), session=cpu_session)
    hand_bbox_detector = KeypointBoundingBoxDetector(
        config=KeypointBoundingBoxDetectorConfig(
            center_keypoint_names=("right_wrist",),
            min_box_size_px=80.0,
        ),
        session=cpu_session,
    )
    hand_stage = DetectionStage(
        name="right_hand",
        keypoint_detectors=[recorder],
        object_detector=hand_bbox_detector,
    )
    body_stage = DetectionStage(
        name="body",
        keypoint_detectors=[
            _WristKeypointDetector(config=_FakeKeypointDetectorConfig(), session=cpu_session)
        ],
        children=[hand_stage],
    )
    return body_stage, recorder


class TestParentKeypointsFlowIntoChildObjectDetector:
    def test_run_crops_child_stage_from_parent_wrist_keypoint(self, cpu_session, image):
        body_stage, recorder = _build_hierarchy(cpu_session)
        obs, _ = body_stage.run(image, StageState())

        assert recorder.seen_crop_shapes == [(80, 80)]
        hand_bboxes = obs.children["right_hand"].bounding_boxes
        assert len(hand_bboxes) == 1
        assert hand_bboxes[0].center == pytest.approx((200.0, 150.0))

    def test_run_batch_crops_child_stage_per_camera_from_parent_wrist_keypoint(
        self, cpu_session, image
    ):
        body_stage, _recorder = _build_hierarchy(cpu_session)
        images = {"cam0": image, "cam1": image}
        obs_per_cam, _ = body_stage.run_batch(images, {})

        for cam_id in ("cam0", "cam1"):
            hand_bboxes = obs_per_cam[cam_id].children["right_hand"].bounding_boxes
            assert len(hand_bboxes) == 1
            assert hand_bboxes[0].center == pytest.approx((200.0, 150.0))
            assert hand_bboxes[0].width == pytest.approx(80.0)
            assert hand_bboxes[0].height == pytest.approx(80.0)

    def test_no_parent_keypoints_means_no_child_bbox(self, cpu_session, image):
        # A body stage with no keypoint detector at all never produces
        # keypoints for the child hand stage to crop from.
        hand_bbox_detector = KeypointBoundingBoxDetector(
            config=KeypointBoundingBoxDetectorConfig(center_keypoint_names=("right_wrist",)),
            session=cpu_session,
        )
        recorder = _RecordingKeypointDetector(config=_FakeKeypointDetectorConfig(), session=cpu_session)
        hand_stage = DetectionStage(
            name="right_hand", keypoint_detectors=[recorder], object_detector=hand_bbox_detector
        )
        body_stage = DetectionStage(name="body", keypoint_detectors=[], children=[hand_stage])

        obs, _ = body_stage.run(image, StageState())

        assert obs.children["right_hand"].bounding_boxes == []
        # No bbox → child falls back to the full (unclipped) parent crop.
        assert recorder.seen_crop_shapes == [image.shape[:2]]


class TestDegenerateChildCropDoesNotCrash:
    """A wrist keypoint near/past the frame edge can produce a fixed-size
    crop box that lands fully outside the image. BoundingBox.clipped()
    collapses that to a zero-area box (rather than raising); DetectionStage
    must then treat the resulting empty crop as "nothing detected" instead
    of handing an empty image to the keypoint detector.
    """

    def _build(self, cpu_session: CpuSession) -> tuple[DetectionStage, _RecordingKeypointDetector]:
        # Wrist far outside a 640x480 frame — the wrist-centered crop box
        # (min_box_size_px=80) lands entirely past the bottom-right edge.
        wrist_detector = _FixedPointKeypointDetector(
            config=_FixedPointKeypointDetectorConfig(point_name="right_wrist", xy=(900.0, 700.0)),
            session=cpu_session,
        )
        recorder = _RecordingKeypointDetector(config=_FakeKeypointDetectorConfig(), session=cpu_session)
        hand_bbox_detector = KeypointBoundingBoxDetector(
            config=KeypointBoundingBoxDetectorConfig(
                center_keypoint_names=("right_wrist",), min_box_size_px=80.0
            ),
            session=cpu_session,
        )
        hand_stage = DetectionStage(
            name="right_hand", keypoint_detectors=[recorder], object_detector=hand_bbox_detector
        )
        body_stage = DetectionStage(
            name="body", keypoint_detectors=[wrist_detector], children=[hand_stage]
        )
        return body_stage, recorder

    def test_run_does_not_crash_and_reports_no_hand_keypoints(self, cpu_session, image):
        body_stage, recorder = self._build(cpu_session)
        obs, _ = body_stage.run(image, StageState())

        hand_obs = obs.children["right_hand"]
        assert hand_obs.keypoints.n_valid == 0
        # Detector itself is never called with an empty image.
        assert recorder.seen_crop_shapes == []

    def test_run_batch_does_not_crash_and_reports_no_hand_keypoints(self, cpu_session, image):
        body_stage, _recorder = self._build(cpu_session)
        obs_per_cam, _ = body_stage.run_batch({"cam0": image}, {})

        hand_obs = obs_per_cam["cam0"].children["right_hand"]
        assert hand_obs.keypoints.n_valid == 0
