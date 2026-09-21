"""Tests for generic object-detector decode utilities: NMS, box-format
conversion, and the pre-NMS / NMS-baked-in decode strategies.
"""

from __future__ import annotations

import numpy as np
import pytest

from skellytracker.core.detectors.processing.object_detection_decode import (
    boxes_to_xyxy,
    decode_nms_baked_in,
    decode_prenms,
    multiclass_nms,
    nms,
)
from skellytracker.core.sidecar.model import DetectionDecodeSpec

_DEFAULT_DECODE_SPEC = DetectionDecodeSpec()

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
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [100.0, 100.0, 110.0, 110.0],
            ]
        )
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores, nms_thr=0.5)
        assert sorted(keep) == [0, 1]

    def test_suppresses_fully_overlapping_lower_score(self):
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
            ]
        )
        scores = np.array([0.9, 0.5])
        keep = nms(boxes, scores, nms_thr=0.5)
        assert keep == [0]

    def test_suppresses_highly_overlapping_box(self):
        boxes = np.array(
            [
                [0.0, 0.0, 100.0, 100.0],
                [1.0, 1.0, 99.0, 99.0],  # nearly same, high IoU
            ]
        )
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores, nms_thr=0.5)
        assert keep == [0]

    def test_keeps_when_iou_below_threshold(self):
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [8.0, 0.0, 20.0, 10.0],  # small overlap
            ]
        )
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores, nms_thr=0.9)
        assert sorted(keep) == [0, 1]

    def test_higher_score_wins(self):
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
            ]
        )
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
# boxes_to_xyxy
# ---------------------------------------------------------------------------


class TestBoxesToXyxy:
    def test_xyxy_is_passthrough(self):
        boxes = np.array([[10.0, 20.0, 50.0, 60.0]])
        result = boxes_to_xyxy(boxes, "xyxy")
        assert np.allclose(result, boxes)

    def test_none_defaults_to_xyxy(self):
        boxes = np.array([[10.0, 20.0, 50.0, 60.0]])
        result = boxes_to_xyxy(boxes, None)
        assert np.allclose(result, boxes)

    def test_xywh_converts_to_xyxy(self):
        boxes = np.array([[10.0, 20.0, 40.0, 30.0]])  # x, y, w, h
        result = boxes_to_xyxy(boxes, "xywh")
        assert np.allclose(result, [[10.0, 20.0, 50.0, 50.0]])

    def test_cxcywh_converts_to_xyxy(self):
        boxes = np.array([[30.0, 35.0, 40.0, 30.0]])  # cx, cy, w, h
        result = boxes_to_xyxy(boxes, "cxcywh")
        assert np.allclose(result, [[10.0, 20.0, 50.0, 50.0]])

    def test_unsupported_format_raises_not_implemented(self):
        boxes = np.array([[10.0, 20.0, 40.0, 30.0]])
        with pytest.raises(NotImplementedError):
            boxes_to_xyxy(boxes, "unsupported_format")


# ---------------------------------------------------------------------------
# decode_nms_baked_in
# ---------------------------------------------------------------------------


class TestDecodeNmsBakedIn:
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
        result_boxes, result_scores = decode_nms_baked_in(
            outputs_one=outputs,
            ratio=1.0,
            score_thr=0.7,
            decode_spec=_DEFAULT_DECODE_SPEC,
        )
        assert len(result_boxes) == 1
        assert result_scores[0] == pytest.approx(0.95)

    def test_filters_low_score_detections(self):
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
        scores = np.array([0.3])
        outputs = self._make_outputs(boxes, scores)
        result_boxes, _ = decode_nms_baked_in(
            outputs_one=outputs,
            ratio=1.0,
            score_thr=0.7,
            decode_spec=_DEFAULT_DECODE_SPEC,
        )
        assert len(result_boxes) == 0

    def test_scales_boxes_by_inverse_ratio(self):
        boxes = np.array([[100.0, 100.0, 200.0, 200.0]])
        scores = np.array([0.95])
        outputs = self._make_outputs(boxes, scores)
        result_boxes, _ = decode_nms_baked_in(
            outputs_one=outputs,
            ratio=2.0,
            score_thr=0.7,
            decode_spec=_DEFAULT_DECODE_SPEC,
        )
        assert len(result_boxes) == 1
        assert result_boxes[0, 0] == pytest.approx(50.0)  # 100 / 2.0

    def test_unsupported_box_format_raises_not_implemented(self):
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
        scores = np.array([0.95])
        outputs = self._make_outputs(boxes, scores)
        decode_spec = DetectionDecodeSpec(box_format="cxcywh")
        result_boxes, _ = decode_nms_baked_in(
            outputs_one=outputs,
            ratio=1.0,
            score_thr=0.7,
            decode_spec=decode_spec,
        )
        # cxcywh conversion is implemented (unlike the prenms/raw-anchor
        # paths) — this just documents that decode_nms_baked_in actually
        # respects decode.box_format rather than assuming xyxy.
        assert len(result_boxes) == 1


# ---------------------------------------------------------------------------
# decode_prenms
# ---------------------------------------------------------------------------


class TestDecodePrenms:
    def test_returns_empty_when_no_scores_pass_threshold(self):
        boxes = np.zeros((1, 5, 4), dtype=np.float32)
        scores = np.full((1, 5), 0.1, dtype=np.float32)
        result_boxes, result_scores = decode_prenms(
            boxes=boxes,
            scores=scores,
            ratio=1.0,
            score_thr=0.7,
            nms_thr=0.45,
            decode_spec=_DEFAULT_DECODE_SPEC,
        )
        assert len(result_boxes) == 0
        assert len(result_scores) == 0

    def test_returns_boxes_above_threshold(self):
        boxes = np.array([[[10.0, 10.0, 50.0, 50.0]]], dtype=np.float32)
        scores = np.array([[0.9]], dtype=np.float32)
        result_boxes, result_scores = decode_prenms(
            boxes=boxes,
            scores=scores,
            ratio=1.0,
            score_thr=0.7,
            nms_thr=0.45,
            decode_spec=_DEFAULT_DECODE_SPEC,
        )
        assert len(result_boxes) == 1

    def test_scales_by_inverse_ratio(self):
        boxes = np.array([[[100.0, 200.0, 300.0, 400.0]]], dtype=np.float32)
        scores = np.array([[0.95]], dtype=np.float32)
        result_boxes, _ = decode_prenms(
            boxes=boxes,
            scores=scores,
            ratio=2.0,
            score_thr=0.7,
            nms_thr=0.45,
            decode_spec=_DEFAULT_DECODE_SPEC,
        )
        assert result_boxes[0, 0] == pytest.approx(50.0)  # 100 / 2.0

    def test_unsupported_box_format_raises_not_implemented(self):
        boxes = np.array([[[10.0, 10.0, 50.0, 50.0]]], dtype=np.float32)
        scores = np.array([[0.9]], dtype=np.float32)
        decode_spec = DetectionDecodeSpec(box_format="cxcywh")
        with pytest.raises(NotImplementedError):
            decode_prenms(
                boxes=boxes,
                scores=scores,
                ratio=1.0,
                score_thr=0.7,
                nms_thr=0.45,
                decode_spec=decode_spec,
            )
