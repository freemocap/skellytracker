"""Tests for OnnxSession.run_batched and the ONNX detector preprocess/postprocess split.

Structure
---------
Model-free tests (no onnxruntime, no network)
    TestYoloxPreprocess     — tensor shape / dtype / metadata
    TestRTMPosePreprocess   — tensor shape / dtype / metadata

Integration tests (require onnxruntime + model download; auto-skip otherwise)
    TestRunBatchedYolox     — run_batched N=1 vs detect(), N=2 key/shape consistency
    TestRunBatchedRTMPose   — run_batched N=2 keypoint shapes
    TestDetectionStageBatch — full stage.run_batch() with YOLOX + RTMPose, 2 cameras
"""
from __future__ import annotations

import numpy as np
import pytest

# Skip the entire module when onnxruntime is not installed.
# YOLOX and RTMPose modules import onnxruntime at load time, so the skip
# must happen before those imports to avoid a collection-time ImportError.
pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

from skellytracker.core.detectors.object_detectors.yolox.yolox_person_detector import (  # noqa: E402
    YoloxPersonDetector,
    YoloxPersonDetectorConfig,
)
from skellytracker.core.detectors.keypoint_detectors.rtmpose.wholebody.rtmpose_wholebody_detector import (  # noqa: E402
    RTMPoseKeypointDetector,
    RTMPoseDetectorConfig,
)
from skellytracker.core.detectors.metadata import YoloxMetadata, RTMPoseMetadata  # noqa: E402

import skellytracker.core.detectors.object_detectors.yolox  # noqa: F401, E402 — registry
import skellytracker.core.detectors.keypoint_detectors.rtmpose  # noqa: F401, E402 — registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_cam(image: np.ndarray) -> dict[str, np.ndarray]:
    return {"cam0": image, "cam1": image.copy()}


# ---------------------------------------------------------------------------
# Model-free: YoloxPersonDetector.preprocess
# ---------------------------------------------------------------------------

class TestYoloxPreprocess:
    @pytest.fixture(scope="class")
    @classmethod
    def detector(cls):
        from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
        session = OnnxSession.create(OnnxSessionConfig(
            batch_size=1,
            models=[YoloxPersonDetector.model_spec("yolox-m")],
        ))
        det = YoloxPersonDetector.create(YoloxPersonDetectorConfig(), session)
        yield det
        session.close()

    def test_tensor_shape_is_3hw(self, detector, test_image):
        tensor, _ = detector.preprocess(test_image)
        h, w = YoloxPersonDetectorConfig().input_size
        assert tensor.shape == (3, h, w)

    def test_tensor_dtype_is_float32(self, detector, test_image):
        tensor, _ = detector.preprocess(test_image)
        assert tensor.dtype == np.float32

    def test_tensor_is_contiguous(self, detector, test_image):
        tensor, _ = detector.preprocess(test_image)
        assert tensor.flags["C_CONTIGUOUS"]

    def test_metadata_type(self, detector, test_image):
        _, meta = detector.preprocess(test_image)
        assert isinstance(meta, YoloxMetadata)

    def test_metadata_ratio_is_positive(self, detector, test_image):
        _, meta = detector.preprocess(test_image)
        assert meta.ratio > 0.0

    def test_metadata_original_size_matches_image(self, detector, test_image):
        _, meta = detector.preprocess(test_image)
        assert meta.original_size == test_image.shape[:2]

    def test_two_identical_images_give_identical_tensors(self, detector, test_image):
        t0, _ = detector.preprocess(test_image)
        t1, _ = detector.preprocess(test_image.copy())
        np.testing.assert_array_equal(t0, t1)


# ---------------------------------------------------------------------------
# Model-free: RTMPoseKeypointDetector.preprocess
# ---------------------------------------------------------------------------

class TestRTMPosePreprocess:
    @pytest.fixture(scope="class")
    @classmethod
    def detector(cls):
        from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
        session = OnnxSession.create(OnnxSessionConfig(
            batch_size=1,
            models=[RTMPoseKeypointDetector.model_spec("rtmw-x-l_256x192")],
        ))
        det = RTMPoseKeypointDetector.create(RTMPoseDetectorConfig(), session)
        yield det
        session.close()

    def test_tensor_shape_is_3hw(self, detector, test_image):
        tensor, _ = detector.preprocess(test_image)
        input_h, input_w = RTMPoseDetectorConfig().input_size
        assert tensor.shape == (3, input_h, input_w)

    def test_tensor_dtype_is_float32(self, detector, test_image):
        tensor, _ = detector.preprocess(test_image)
        assert tensor.dtype == np.float32

    def test_tensor_is_contiguous(self, detector, test_image):
        tensor, _ = detector.preprocess(test_image)
        assert tensor.flags["C_CONTIGUOUS"]

    def test_metadata_type(self, detector, test_image):
        _, meta = detector.preprocess(test_image)
        assert isinstance(meta, RTMPoseMetadata)

    def test_metadata_center_shape(self, detector, test_image):
        _, meta = detector.preprocess(test_image)
        assert meta.center.shape == (2,)

    def test_metadata_scale_shape(self, detector, test_image):
        _, meta = detector.preprocess(test_image)
        assert meta.scale.shape == (2,)


# ---------------------------------------------------------------------------
# Integration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def yolox_session_n1():
    from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
    cfg = OnnxSessionConfig(
        batch_size=1,
        models=[YoloxPersonDetector.model_spec("yolox-m")],
    )
    session = OnnxSession.create(cfg)
    yield session
    session.close()


@pytest.fixture(scope="module")
def yolox_session_n2():
    from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
    cfg = OnnxSessionConfig(
        batch_size=2,
        models=[YoloxPersonDetector.model_spec("yolox-m")],
    )
    session = OnnxSession.create(cfg)
    yield session
    session.close()


@pytest.fixture(scope="module")
def rtmpose_session_n2():
    from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
    cfg = OnnxSessionConfig(
        batch_size=2,
        models=[RTMPoseKeypointDetector.model_spec("rtmw-x-l_256x192")],
    )
    session = OnnxSession.create(cfg)
    yield session
    session.close()


@pytest.fixture(scope="module")
def combined_session_n2():
    """Single session holding both YOLOX and RTMPose at batch_size=2."""
    from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
    cfg = OnnxSessionConfig(
        batch_size=2,
        models=[
            YoloxPersonDetector.model_spec("yolox-m"),
            RTMPoseKeypointDetector.model_spec("rtmw-x-l_256x192"),
        ],
    )
    session = OnnxSession.create(cfg)
    yield session
    session.close()


# ---------------------------------------------------------------------------
# TestRunBatchedYolox
# ---------------------------------------------------------------------------

class TestRunBatchedYolox:
    def test_single_camera_keys_preserved(self, test_image, yolox_session_n1):
        detector = YoloxPersonDetector.create(YoloxPersonDetectorConfig(), yolox_session_n1)
        tensor, _ = detector.preprocess(test_image)
        result = yolox_session_n1.run_batched("yolox-m", {"cam0": tensor})
        assert list(result.keys()) == ["cam0"]

    def test_single_camera_matches_detect(self, test_image, yolox_session_n1):
        """run_batched N=1 must give the same detections as detect()."""
        config = YoloxPersonDetectorConfig()
        detector = YoloxPersonDetector.create(config, yolox_session_n1)
        single = detector.detect(test_image)

        tensor, meta = detector.preprocess(test_image)
        raw_batch = yolox_session_n1.run_batched("yolox-m", {"cam0": tensor})
        batched = detector.postprocess(raw_batch["cam0"], meta)

        assert len(single) == len(batched)
        for s, b in zip(single, batched, strict=True):
            assert s.x1 == pytest.approx(b.x1, abs=1.0)
            assert s.y1 == pytest.approx(b.y1, abs=1.0)
            assert s.confidence == pytest.approx(b.confidence, abs=1e-4)

    def test_two_cameras_keys_preserved(self, test_image, yolox_session_n2):
        detector = YoloxPersonDetector.create(YoloxPersonDetectorConfig(), yolox_session_n2)
        tensors = {k: detector.preprocess(img)[0] for k, img in _two_cam(test_image).items()}
        result = yolox_session_n2.run_batched("yolox-m", tensors)
        assert set(result.keys()) == {"cam0", "cam1"}

    def test_two_cameras_output_is_list_per_camera(self, test_image, yolox_session_n2):
        detector = YoloxPersonDetector.create(YoloxPersonDetectorConfig(), yolox_session_n2)
        tensors = {k: detector.preprocess(img)[0] for k, img in _two_cam(test_image).items()}
        result = yolox_session_n2.run_batched("yolox-m", tensors)
        assert isinstance(result["cam0"], list)
        assert isinstance(result["cam1"], list)

    def test_two_identical_cameras_give_identical_detections(self, test_image, yolox_session_n2):
        config = YoloxPersonDetectorConfig()
        detector = YoloxPersonDetector.create(config, yolox_session_n2)
        images = _two_cam(test_image)
        preprocessed = {k: detector.preprocess(img) for k, img in images.items()}
        tensors = {k: v[0] for k, v in preprocessed.items()}
        metas = {k: v[1] for k, v in preprocessed.items()}

        raw = yolox_session_n2.run_batched("yolox-m", tensors)
        boxes0 = detector.postprocess(raw["cam0"], metas["cam0"])
        boxes1 = detector.postprocess(raw["cam1"], metas["cam1"])

        assert len(boxes0) == len(boxes1)
        for b0, b1 in zip(boxes0, boxes1, strict=True):
            assert b0.x1 == pytest.approx(b1.x1, abs=1.0)
            assert b0.confidence == pytest.approx(b1.confidence, abs=1e-4)

    def test_two_cameras_detect_person_in_both(self, test_image, yolox_session_n2):
        config = YoloxPersonDetectorConfig()
        detector = YoloxPersonDetector.create(config, yolox_session_n2)
        images = _two_cam(test_image)
        preprocessed = {k: detector.preprocess(img) for k, img in images.items()}
        tensors = {k: v[0] for k, v in preprocessed.items()}
        metas = {k: v[1] for k, v in preprocessed.items()}

        raw = yolox_session_n2.run_batched("yolox-m", tensors)
        for cam_id in ("cam0", "cam1"):
            boxes = detector.postprocess(raw[cam_id], metas[cam_id])
            assert len(boxes) > 0, f"Expected a person detected in {cam_id}"


# ---------------------------------------------------------------------------
# TestRunBatchedRTMPose
# ---------------------------------------------------------------------------

class TestRunBatchedRTMPose:
    def test_two_cameras_keys_preserved(self, test_image, rtmpose_session_n2):
        detector = RTMPoseKeypointDetector.create(RTMPoseDetectorConfig(), rtmpose_session_n2)
        tensors = {k: detector.preprocess(img)[0] for k, img in _two_cam(test_image).items()}
        result = rtmpose_session_n2.run_batched("rtmw-x-l_256x192", tensors)
        assert set(result.keys()) == {"cam0", "cam1"}

    def test_two_cameras_give_133_keypoints_each(self, test_image, rtmpose_session_n2):
        config = RTMPoseDetectorConfig()
        detector = RTMPoseKeypointDetector.create(config, rtmpose_session_n2)
        images = _two_cam(test_image)
        preprocessed = {k: detector.preprocess(img) for k, img in images.items()}
        tensors = {k: v[0] for k, v in preprocessed.items()}
        metas = {k: v[1] for k, v in preprocessed.items()}

        raw = rtmpose_session_n2.run_batched("rtmw-x-l_256x192", tensors)
        for cam_id in ("cam0", "cam1"):
            kpts = detector.postprocess(raw[cam_id], metas[cam_id])
            assert kpts.xyz.shape == (133, 3), f"{cam_id}: expected (133, 3)"

    def test_two_identical_cameras_give_consistent_keypoint_shapes(self, test_image, rtmpose_session_n2):
        detector = RTMPoseKeypointDetector.create(RTMPoseDetectorConfig(), rtmpose_session_n2)
        images = _two_cam(test_image)
        preprocessed = {k: detector.preprocess(img) for k, img in images.items()}
        tensors = {k: v[0] for k, v in preprocessed.items()}
        metas = {k: v[1] for k, v in preprocessed.items()}

        raw = rtmpose_session_n2.run_batched("rtmw-x-l_256x192", tensors)
        kpts0 = detector.postprocess(raw["cam0"], metas["cam0"])
        kpts1 = detector.postprocess(raw["cam1"], metas["cam1"])
        assert kpts0.xyz.shape == kpts1.xyz.shape
        assert kpts0.names == kpts1.names

    def test_single_camera_batched_matches_detect_shape(self, test_image, rtmpose_session_n2):
        """N=1 batch gives same keypoint shape as detect()."""
        config = RTMPoseDetectorConfig()
        detector = RTMPoseKeypointDetector.create(config, rtmpose_session_n2)
        single = detector.detect(test_image)

        tensor, meta = detector.preprocess(test_image)
        raw = rtmpose_session_n2.run_batched("rtmw-x-l_256x192", {"cam0": tensor})
        batched = detector.postprocess(raw["cam0"], meta)

        assert batched.xyz.shape == single.xyz.shape
        assert batched.names == single.names


# ---------------------------------------------------------------------------
# TestDetectionStageBatch — full YOLOX + RTMPose pipeline
# ---------------------------------------------------------------------------

class TestDetectionStageBatch:
    @pytest.fixture(scope="class")
    @classmethod
    def stage_and_session(cls, combined_session_n2):
        from skellytracker.core import DetectionStageConfig, TrackerConfig, Tracker
        from skellytracker.core.tracker.detection_stage import DetectionStage
        from skellytracker.core.config.detection_stage_config import DetectionStageConfig as StageConfig
        from skellytracker.core.detectors.detector_base_classes import (
            build_keypoint_detector,
            build_object_detector,
        )
        from skellytracker.core.temporal_processing.bbox_policy import BBoxPolicy

        stage = DetectionStage.create(
            StageConfig(
                name="body",
                object_detector=YoloxPersonDetectorConfig(),
                keypoint_detectors=[RTMPoseDetectorConfig()],
            ),
            sessions={"onnx": combined_session_n2},
        )
        return stage, combined_session_n2

    def test_run_batch_returns_obs_per_camera(self, stage_and_session, test_image):
        from skellytracker.core.tracker.tracker_state import StageState
        from skellytracker.core.detectors.detection_context import DetectionContext

        stage, _ = stage_and_session
        images = _two_cam(test_image)
        states = {"cam0": StageState(), "cam1": StageState()}
        ctx = DetectionContext(frame_number=0, timestamp_ms=0)

        obs_batch, states_batch = stage.run_batch(images, states, ctx)
        assert set(obs_batch.keys()) == {"cam0", "cam1"}
        assert set(states_batch.keys()) == {"cam0", "cam1"}

    def test_run_batch_keypoints_133_per_camera(self, stage_and_session, test_image):
        from skellytracker.core.tracker.tracker_state import StageState
        from skellytracker.core.detectors.detection_context import DetectionContext

        stage, _ = stage_and_session
        images = _two_cam(test_image)
        states = {"cam0": StageState(), "cam1": StageState()}
        ctx = DetectionContext(frame_number=0, timestamp_ms=0)

        obs_batch, _ = stage.run_batch(images, states, ctx)
        for cam_id in ("cam0", "cam1"):
            kpts = obs_batch[cam_id].keypoints
            assert kpts is not None, f"{cam_id}: keypoints is None"
            assert kpts.xyz.shape == (133, 3), f"{cam_id}: expected (133, 3)"

    def test_run_batch_states_are_independent(self, stage_and_session, test_image):
        from skellytracker.core.tracker.tracker_state import StageState
        from skellytracker.core.detectors.detection_context import DetectionContext

        stage, _ = stage_and_session
        images = _two_cam(test_image)
        states = {"cam0": StageState(), "cam1": StageState()}
        ctx = DetectionContext(frame_number=0, timestamp_ms=0)

        _, states_batch = stage.run_batch(images, states, ctx)
        assert states_batch["cam0"] is not states_batch["cam1"]
