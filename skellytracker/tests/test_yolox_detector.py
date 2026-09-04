"""Tests for YOLOX preprocessing, NMS, postprocessing, config, and detector construction.

Model-free tests (no ONNX runtime or network needed) cover preprocessing, NMS, postprocessing,
config, model_spec, and create type guards.

Integration tests at the bottom (TestYoloxInference) require onnxruntime and a network connection
to download the model. They are skipped automatically when onnxruntime is not installed.
"""
from __future__ import annotations

import numpy as np
import pytest

from skellytracker.core.detectors.object_detectors.yolox.yolox_person_detector import (
    YoloxPersonDetector,
    YoloxPersonDetectorConfig,
    _postprocess_prenms,
    _postprocess_yolox,
)
from skellytracker.core.detectors.object_detectors.yolox.yolox_preprocessing import (
    multiclass_nms,
    nms,
    yolox_letterbox_preprocess,
)
from skellytracker.core.sessions.cpu_session import CpuSession, CpuSessionConfig
from skellytracker.core.sessions.onnx_session import OnnxModelSpec


# ---------------------------------------------------------------------------
# yolox_letterbox_preprocess
# ---------------------------------------------------------------------------

class TestLetterboxPreprocess:
    def test_output_shape_matches_target(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, _ = yolox_letterbox_preprocess(img, (640, 640))
        assert padded.shape == (640, 640, 3)

    def test_ratio_for_square_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, ratio = yolox_letterbox_preprocess(img, (200, 200))
        assert ratio == pytest.approx(2.0)

    def test_ratio_limited_by_shorter_side(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, ratio = yolox_letterbox_preprocess(img, (640, 640))
        assert ratio == pytest.approx(640 / 640)

    def test_ratio_limited_by_height(self):
        img = np.zeros((800, 400, 3), dtype=np.uint8)
        _, ratio = yolox_letterbox_preprocess(img, (640, 640))
        assert ratio == pytest.approx(640 / 800)

    def test_padding_color_is_114(self):
        # 100h × 50w image → ratio=2.0, resized to 200h × 100w.
        # Columns 100–200 are untouched padding (value 114).
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        padded, _ = yolox_letterbox_preprocess(img, (200, 200))
        assert padded[100, 150, 0] == 114

    def test_output_dtype_is_uint8(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        padded, _ = yolox_letterbox_preprocess(img, (416, 416))
        assert padded.dtype == np.uint8


# ---------------------------------------------------------------------------
# nms
# ---------------------------------------------------------------------------

class TestNms:
    def test_keeps_single_box(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        scores = np.array([0.9])
        keep = nms(boxes, scores, nms_thr=0.5)
        assert keep == [0]

    def test_keeps_non_overlapping_boxes(self):
        boxes = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [100.0, 100.0, 110.0, 110.0],
        ])
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores, nms_thr=0.5)
        assert sorted(keep) == [0, 1]

    def test_suppresses_fully_overlapping_lower_score(self):
        boxes = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ])
        scores = np.array([0.9, 0.5])
        keep = nms(boxes, scores, nms_thr=0.5)
        assert keep == [0]

    def test_suppresses_highly_overlapping_box(self):
        boxes = np.array([
            [0.0, 0.0, 100.0, 100.0],
            [1.0, 1.0, 99.0, 99.0],  # nearly same, high IoU
        ])
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores, nms_thr=0.5)
        assert keep == [0]

    def test_keeps_when_iou_below_threshold(self):
        boxes = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [8.0, 0.0, 20.0, 10.0],  # small overlap
        ])
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores, nms_thr=0.9)
        assert sorted(keep) == [0, 1]

    def test_higher_score_wins(self):
        boxes = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ])
        scores = np.array([0.4, 0.9])
        keep = nms(boxes, scores, nms_thr=0.5)
        assert keep == [1]


# ---------------------------------------------------------------------------
# multiclass_nms
# ---------------------------------------------------------------------------

class TestMulticlassNms:
    def test_returns_none_when_no_valid_scores(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        scores = np.array([[0.1]])  # below any reasonable threshold
        dets, keep = multiclass_nms(boxes, scores, nms_thr=0.5, score_thr=0.5)
        assert dets is None
        assert keep is None

    def test_output_has_6_columns(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        scores = np.array([[0.9]])
        dets, _ = multiclass_nms(boxes, scores, nms_thr=0.5, score_thr=0.5)
        assert dets is not None
        assert dets.shape[1] == 6

    def test_keeps_high_score_box(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 200.0, 200.0]])
        scores = np.array([[0.9], [0.3]])
        dets, _ = multiclass_nms(boxes, scores, nms_thr=0.5, score_thr=0.5)
        assert dets is not None
        assert len(dets) == 1
        assert dets[0, 4] == pytest.approx(0.9)

    def test_class_index_recorded_in_column_5(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        scores = np.array([[0.0, 0.95]])  # class 1 wins
        dets, _ = multiclass_nms(boxes, scores, nms_thr=0.5, score_thr=0.5)
        assert dets is not None
        assert dets[0, 5] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _postprocess_yolox — baked-NMS format (last dim == 5)
# ---------------------------------------------------------------------------

class TestPostprocessYoloxBakedNms:
    def _make_outputs(self, boxes_xyxy: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Pack boxes + scores into the (1, N, 5) baked-NMS format."""
        n = len(boxes_xyxy)
        out = np.zeros((1, n, 5), dtype=np.float32)
        out[0, :, :4] = boxes_xyxy
        out[0, :, 4] = scores
        return out

    def test_returns_boxes_above_threshold(self):
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
        scores = np.array([0.95])
        outputs = self._make_outputs(boxes, scores)
        result_boxes, result_scores = _postprocess_yolox(
            outputs_one=outputs, ratio=1.0,
            model_input_size=(640, 640),
            score_thr=0.7, nms_thr=0.45,
        )
        assert len(result_boxes) == 1
        assert result_scores[0] == pytest.approx(0.95)

    def test_filters_low_score_detections(self):
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
        scores = np.array([0.3])
        outputs = self._make_outputs(boxes, scores)
        result_boxes, _ = _postprocess_yolox(
            outputs_one=outputs, ratio=1.0,
            model_input_size=(640, 640),
            score_thr=0.7, nms_thr=0.45,
        )
        assert len(result_boxes) == 0

    def test_scales_boxes_by_inverse_ratio(self):
        boxes = np.array([[100.0, 100.0, 200.0, 200.0]])
        scores = np.array([0.95])
        outputs = self._make_outputs(boxes, scores)
        result_boxes, _ = _postprocess_yolox(
            outputs_one=outputs, ratio=2.0,
            model_input_size=(640, 640),
            score_thr=0.7, nms_thr=0.45,
        )
        assert len(result_boxes) == 1
        assert result_boxes[0, 0] == pytest.approx(50.0)  # 100 / 2.0

    def test_raises_on_unexpected_output_shape(self):
        bad = np.zeros((1, 10, 7), dtype=np.float32)
        with pytest.raises(RuntimeError, match="Unexpected YOLOX output shape"):
            _postprocess_yolox(
                outputs_one=bad, ratio=1.0,
                model_input_size=(640, 640),
                score_thr=0.7, nms_thr=0.45,
            )


# ---------------------------------------------------------------------------
# _postprocess_prenms
# ---------------------------------------------------------------------------

class TestPostprocessPrenms:
    def test_returns_empty_when_no_scores_pass_threshold(self):
        boxes = np.zeros((1, 5, 4), dtype=np.float32)
        scores = np.full((1, 5), 0.1, dtype=np.float32)
        result_boxes, result_scores = _postprocess_prenms(
            boxes=boxes, scores=scores, ratio=1.0, score_thr=0.7, nms_thr=0.45
        )
        assert len(result_boxes) == 0
        assert len(result_scores) == 0

    def test_returns_boxes_above_threshold(self):
        boxes = np.array([[[10.0, 10.0, 50.0, 50.0]]], dtype=np.float32)
        scores = np.array([[0.9]], dtype=np.float32)
        result_boxes, result_scores = _postprocess_prenms(
            boxes=boxes, scores=scores, ratio=1.0, score_thr=0.7, nms_thr=0.45
        )
        assert len(result_boxes) == 1

    def test_scales_by_inverse_ratio(self):
        boxes = np.array([[[100.0, 200.0, 300.0, 400.0]]], dtype=np.float32)
        scores = np.array([[0.95]], dtype=np.float32)
        result_boxes, _ = _postprocess_prenms(
            boxes=boxes, scores=scores, ratio=2.0, score_thr=0.7, nms_thr=0.45
        )
        assert result_boxes[0, 0] == pytest.approx(50.0)  # 100 / 2.0


# ---------------------------------------------------------------------------
# YoloxPersonDetectorConfig
# ---------------------------------------------------------------------------

class TestYoloxPersonDetectorConfig:
    def test_input_size_for_yolox_m(self):
        config = YoloxPersonDetectorConfig(model_name="yolox-m")
        assert config.input_size == (640, 640)

    def test_input_size_for_yolox_tiny(self):
        config = YoloxPersonDetectorConfig(model_name="yolox-tiny")
        assert config.input_size == (416, 416)

    def test_input_size_fallback_for_unknown_model(self):
        config = YoloxPersonDetectorConfig(model_name="nonexistent-variant")
        assert config.input_size == (640, 640)

    def test_max_detections_none_means_unlimited(self):
        config = YoloxPersonDetectorConfig(max_detections=None)
        assert config.max_detections is None


# ---------------------------------------------------------------------------
# YoloxPersonDetector.model_spec
# ---------------------------------------------------------------------------

class TestYoloxPersonDetectorModelSpec:
    def test_returns_onnx_model_spec_for_known_model(self):
        spec = YoloxPersonDetector.model_spec("yolox-m")
        assert isinstance(spec, OnnxModelSpec)
        assert spec.name == "yolox-m"
        assert spec.input_size == (640, 640)

    def test_returns_onnx_model_spec_for_tiny(self):
        spec = YoloxPersonDetector.model_spec("yolox-tiny")
        assert spec.name == "yolox-tiny"
        assert spec.input_size == (416, 416)

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown YOLOX model"):
            YoloxPersonDetector.model_spec("yolox-xl")


# ---------------------------------------------------------------------------
# YoloxPersonDetector.create — type-guard checks (no model needed)
# ---------------------------------------------------------------------------

class TestYoloxPersonDetectorCreate:
    def test_wrong_session_type_raises_type_error(self):
        config = YoloxPersonDetectorConfig()
        bad_session = CpuSession.create(CpuSessionConfig())
        with pytest.raises(TypeError, match="OnnxSession"):
            YoloxPersonDetector.create(config, bad_session)
        bad_session.close()

    def test_wrong_config_type_raises_type_error(self):
        pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
        from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
        from skellytracker.core.detectors.keypoint_detectors.rtmpose import RTMPoseKeypointDetector
        session_config = OnnxSessionConfig(
            batch_size=1,
            models=[RTMPoseKeypointDetector.model_spec("rtmw-x-l_256x192")],
        )
        session = OnnxSession.create(session_config)
        try:
            bad_config = CpuSessionConfig()
            with pytest.raises((TypeError, Exception)):
                YoloxPersonDetector.create(bad_config, session)
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Integration — model download + real inference
# (requires onnxruntime and network; skipped automatically otherwise)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def yolox_onnx_session():
    pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
    from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
    import skellytracker.core.detectors.object_detectors.yolox  # noqa: F401

    config = OnnxSessionConfig(
        batch_size=1,
        models=[YoloxPersonDetector.model_spec("yolox-m")],
    )
    session = OnnxSession.create(config)
    yield session
    session.close()


class TestYoloxInference:
    def test_session_loads_yolox_model(self, yolox_onnx_session):
        assert yolox_onnx_session.get_session("yolox-m") is not None

    def test_detect_returns_list(self, test_image, yolox_onnx_session):
        detector = YoloxPersonDetector.create(YoloxPersonDetectorConfig(), yolox_onnx_session)
        result = detector.detect(test_image)
        assert isinstance(result, list)

    def test_detect_finds_person_in_test_image(self, test_image, yolox_onnx_session):
        detector = YoloxPersonDetector.create(YoloxPersonDetectorConfig(), yolox_onnx_session)
        result = detector.detect(test_image)
        assert len(result) > 0, "Expected at least one person in the test image"

    def test_detect_returns_valid_bounding_boxes(self, test_image, yolox_onnx_session):
        detector = YoloxPersonDetector.create(YoloxPersonDetectorConfig(), yolox_onnx_session)
        for bb in detector.detect(test_image):
            assert bb.x1 < bb.x2
            assert bb.y1 < bb.y2
            assert 0.0 <= bb.confidence <= 1.0

    def test_results_sorted_by_confidence_descending(self, test_image, yolox_onnx_session):
        config = YoloxPersonDetectorConfig(max_detections=None)
        detector = YoloxPersonDetector.create(config, yolox_onnx_session)
        result = detector.detect(test_image)
        scores = [bb.confidence for bb in result]
        assert scores == sorted(scores, reverse=True)

    def test_max_detections_limits_results(self, test_image, yolox_onnx_session):
        config = YoloxPersonDetectorConfig(max_detections=1)
        detector = YoloxPersonDetector.create(config, yolox_onnx_session)
        assert len(detector.detect(test_image)) <= 1

    def test_max_detections_none_returns_all(self, test_image, yolox_onnx_session):
        config_limited = YoloxPersonDetectorConfig(max_detections=1)
        config_all = YoloxPersonDetectorConfig(max_detections=None)
        detector_limited = YoloxPersonDetector.create(config_limited, yolox_onnx_session)
        detector_all = YoloxPersonDetector.create(config_all, yolox_onnx_session)
        assert len(detector_all.detect(test_image)) >= len(detector_limited.detect(test_image))

    def test_blank_image_returns_empty_list(self, yolox_onnx_session):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = YoloxPersonDetector.create(YoloxPersonDetectorConfig(), yolox_onnx_session)
        result = detector.detect(blank)
        assert isinstance(result, list)
        assert len(result) == 0
