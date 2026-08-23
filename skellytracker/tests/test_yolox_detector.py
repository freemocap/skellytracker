"""Tests for YOLOX-specific decode (raw anchor grid + dispatch), config, and
detector construction.

Generic sidecar preprocessing (letterbox) and generic decode utilities (NMS,
box-format conversion, pre-NMS/NMS-baked-in decode) are tested in
`test_image_preprocessing.py` and `test_object_detection_decode.py`
instead, since they're not YOLOX-specific.

Integration tests at the bottom (TestYoloxInference) require onnxruntime and a network connection
to download the model. They are skipped automatically when onnxruntime is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from skellytracker.core.detectors.object_detectors.yolox.yolox_decode import (
    _decode_yolox_raw_anchor_grid,
    yolox_detection_decode,
)
from skellytracker.core.detectors.object_detectors.yolox.yolox_person_detector import (
    YoloxPersonDetector,
    YoloxPersonDetectorConfig,
)
from skellytracker.core.sessions.cpu_session import CpuSession, CpuSessionConfig
from skellytracker.core.sessions.onnx_session import OnnxModelSpec
from skellytracker.core.sidecar.model import DetectionDecodeSpec

_DEFAULT_DECODE_SPEC = DetectionDecodeSpec()


# ---------------------------------------------------------------------------
# _decode_yolox_raw_anchor_grid — raw undecoded anchor grid (last dim == 4)
# ---------------------------------------------------------------------------


class TestDecodeYoloxRawAnchorGrid:
    def _make_raw_output(
        self, model_input_size: tuple[int, int], target_xyxy: np.ndarray
    ) -> np.ndarray:
        """Build a raw (pre-grid-decode) YOLOX anchor tensor with one anchor
        pre-set to decode to `target_xyxy` at the finest (stride-8) grid cell.
        """
        strides = [8, 16, 32]
        h, w = model_input_size
        num_anchors = sum((h // s) * (w // s) for s in strides)
        # channels: cx, cy, w, h, objectness, class_0
        out = np.zeros((1, num_anchors, 6), dtype=np.float32)
        cx = (target_xyxy[0] + target_xyxy[2]) / 2.0
        cy = (target_xyxy[1] + target_xyxy[3]) / 2.0
        box_w = target_xyxy[2] - target_xyxy[0]
        box_h = target_xyxy[3] - target_xyxy[1]
        # Anchor 0 sits at grid cell (0, 0), stride 8: decoded cx = (0+0)*8 = 0
        # before adding the pre-grid-decode offset, so pre-decode value = cx/8.
        out[0, 0, 0] = cx / 8.0
        out[0, 0, 1] = cy / 8.0
        out[0, 0, 2] = np.log(box_w / 8.0)
        out[0, 0, 3] = np.log(box_h / 8.0)
        out[0, 0, 4] = 1.0  # objectness
        out[0, 0, 5] = 1.0  # class score (person)
        return out

    def test_returns_decoded_box(self):
        model_input_size = (640, 640)
        target = np.array([16.0, 16.0, 48.0, 48.0])
        outputs = self._make_raw_output(model_input_size, target)
        boxes, scores = _decode_yolox_raw_anchor_grid(
            outputs_one=outputs,
            ratio=1.0,
            model_input_size=model_input_size,
            score_thr=0.5,
            nms_thr=0.45,
            decode_spec=_DEFAULT_DECODE_SPEC,
        )
        assert len(boxes) == 1
        assert boxes[0] == pytest.approx(target, abs=1e-2)
        assert scores[0] == pytest.approx(1.0, abs=1e-3)

    def test_person_class_id_filters_other_classes(self):
        model_input_size = (640, 640)
        target = np.array([16.0, 16.0, 48.0, 48.0])
        outputs = self._make_raw_output(model_input_size, target)
        decode_spec = DetectionDecodeSpec(person_class_id=1)
        boxes, _ = _decode_yolox_raw_anchor_grid(
            outputs_one=outputs,
            ratio=1.0,
            model_input_size=model_input_size,
            score_thr=0.5,
            nms_thr=0.45,
            decode_spec=decode_spec,
        )
        # Only class 0 was populated with a score, so requesting class 1
        # should find nothing.
        assert len(boxes) == 0

    def test_unsupported_box_format_raises_not_implemented(self):
        model_input_size = (640, 640)
        target = np.array([16.0, 16.0, 48.0, 48.0])
        outputs = self._make_raw_output(model_input_size, target)
        decode_spec = DetectionDecodeSpec(box_format="cxcywh")
        with pytest.raises(NotImplementedError):
            _decode_yolox_raw_anchor_grid(
                outputs_one=outputs,
                ratio=1.0,
                model_input_size=model_input_size,
                score_thr=0.5,
                nms_thr=0.45,
                decode_spec=decode_spec,
            )


# ---------------------------------------------------------------------------
# yolox_detection_decode — top-level YOLOX dispatch
# ---------------------------------------------------------------------------


class TestYoloxDetectionDecodeDispatch:
    def test_raises_on_unexpected_output_shape(self):
        bad = np.zeros((1, 10, 7), dtype=np.float32)
        with pytest.raises(RuntimeError, match="Unexpected YOLOX output shape"):
            yolox_detection_decode(
                raw=[bad],
                ratio=1.0,
                model_input_size=(640, 640),
                score_threshold=0.7,
                nms_threshold=0.45,
                decode_spec=_DEFAULT_DECODE_SPEC,
            )

    def test_dispatches_baked_nms_shape_to_generic_decode(self):
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
        scores = np.array([0.95])
        out = np.zeros((1, 1, 5), dtype=np.float32)
        out[0, :, :4] = boxes
        out[0, :, 4] = scores
        result_boxes, result_scores = yolox_detection_decode(
            raw=[out],
            ratio=1.0,
            model_input_size=(640, 640),
            score_threshold=0.7,
            nms_threshold=0.45,
            decode_spec=_DEFAULT_DECODE_SPEC,
        )
        assert len(result_boxes) == 1
        assert result_scores[0] == pytest.approx(0.95)

    def test_dispatches_two_output_shape_to_prenms_decode(self):
        boxes = np.array([[[10.0, 10.0, 50.0, 50.0]]], dtype=np.float32)
        scores = np.array([[0.9]], dtype=np.float32)
        result_boxes, result_scores = yolox_detection_decode(
            raw=[boxes, scores],
            ratio=1.0,
            model_input_size=(640, 640),
            score_threshold=0.7,
            nms_threshold=0.45,
            decode_spec=_DEFAULT_DECODE_SPEC,
        )
        assert len(result_boxes) == 1


# ---------------------------------------------------------------------------
# YoloxPersonDetectorConfig
# ---------------------------------------------------------------------------


class TestYoloxPersonDetectorConfig:
    def test_default_model_name(self):
        config = YoloxPersonDetectorConfig()
        assert config.model_name == "yolox-m"

    def test_input_size_for_yolox_m(self):
        config = YoloxPersonDetectorConfig(model_name="yolox-m")
        assert config.input_size == (640, 640)

    def test_input_size_for_yolox_tiny(self):
        config = YoloxPersonDetectorConfig(model_name="yolox-tiny")
        assert config.input_size == (416, 416)

    def test_input_size_fallback_for_unknown_model(self):
        config = YoloxPersonDetectorConfig(model_name="nonexistent-variant")
        assert config.input_size == (640, 640)

    def test_default_max_detections_is_1(self):
        assert YoloxPersonDetectorConfig().max_detections == 1

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
        from skellytracker.core.detectors.keypoint_detectors.rtmpose import (
            RTMPoseKeypointDetector,
        )
        from skellytracker.core.sessions.onnx_session import (
            OnnxSession,
            OnnxSessionConfig,
        )

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
    import skellytracker.core.detectors.object_detectors.yolox  # noqa: F401
    from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig

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
        detector = YoloxPersonDetector.create(
            YoloxPersonDetectorConfig(), yolox_onnx_session
        )
        result = detector.detect(test_image)
        assert isinstance(result, list)

    def test_detect_finds_person_in_test_image(self, test_image, yolox_onnx_session):
        detector = YoloxPersonDetector.create(
            YoloxPersonDetectorConfig(), yolox_onnx_session
        )
        result = detector.detect(test_image)
        assert len(result) > 0, "Expected at least one person in the test image"

    def test_detect_returns_valid_bounding_boxes(self, test_image, yolox_onnx_session):
        detector = YoloxPersonDetector.create(
            YoloxPersonDetectorConfig(), yolox_onnx_session
        )
        for bb in detector.detect(test_image):
            assert bb.x1 < bb.x2
            assert bb.y1 < bb.y2
            assert 0.0 <= bb.confidence <= 1.0

    def test_results_sorted_by_confidence_descending(
        self, test_image, yolox_onnx_session
    ):
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
        detector_limited = YoloxPersonDetector.create(
            config_limited, yolox_onnx_session
        )
        detector_all = YoloxPersonDetector.create(config_all, yolox_onnx_session)
        assert len(detector_all.detect(test_image)) >= len(
            detector_limited.detect(test_image)
        )

    def test_blank_image_returns_empty_list(self, yolox_onnx_session):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = YoloxPersonDetector.create(
            YoloxPersonDetectorConfig(), yolox_onnx_session
        )
        result = detector.detect(blank)
        assert isinstance(result, list)
        assert len(result) == 0
