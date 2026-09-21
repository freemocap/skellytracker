"""YOLOX-specific detection decode: raw anchor-grid reconstruction.

Vendored from rtmlib (``tools/pose_estimation/rtmo.py``). Image preprocessing
(letterbox/normalize) and the decode strategies that don't need YOLOX-specific
math (NMS, box-format conversion, pre-NMS/baked-NMS decode) live in
`skellytracker/core/detectors/processing/image_preprocessing.py` and
`skellytracker/core/detectors/processing/object_detection_decode.py` — this file only
keeps the YOLOX anchor-grid decode (stride-based grid reconstruction unique
to YOLOX's head) and the top-level dispatcher that picks between all of them
based on what a given YOLOX ONNX export actually emits.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.detectors.processing.object_detection_decode import (
    decode_nms_baked_in,
    decode_prenms,
    multiclass_nms,
)
from skellytracker.core.sidecar.model import DetectionDecodeSpec


def _decode_yolox_raw_anchor_grid(
    outputs_one: NDArray,
    ratio: float,
    model_input_size: tuple[int, int],
    score_thr: float,
    nms_thr: float,
    decode_spec: DetectionDecodeSpec,
) -> tuple[NDArray, NDArray]:
    """Decode YOLOX's raw (undecoded, un-NMS'd) anchor grid output.

    `outputs_one` is `(1, num_anchors, C)` where `C` encodes cxcywh +
    objectness + per-class scores, laid out per YOLOX's anchor-free head with
    strides `[8, 16, 32]`. This grid reconstruction (and the cxcywh encoding
    it decodes) is specific to that head design — not a generalizable decode
    strategy — so it lives here rather than in
    `core/detectors/object_detection_decode.py`.
    """
    if decode_spec.box_format not in (None, "xyxy"):
        raise NotImplementedError(
            "yolox raw-anchor decode only supports box_format='xyxy' output, "
            f"got {decode_spec.box_format!r}"
        )

    grids: list[NDArray] = []
    expanded_strides: list[NDArray] = []
    strides = [8, 16, 32]
    hsizes = [model_input_size[0] // s for s in strides]
    wsizes = [model_input_size[1] // s for s in strides]
    for hsize, wsize, stride in zip(hsizes, wsizes, strides, strict=False):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))
    grids_ = np.concatenate(grids, 1)
    expanded_strides_ = np.concatenate(expanded_strides, 1)
    outputs_one[..., :2] = (outputs_one[..., :2] + grids_) * expanded_strides_
    outputs_one[..., 2:4] = np.exp(outputs_one[..., 2:4]) * expanded_strides_

    predictions = outputs_one[0]
    boxes_cxcy = predictions[:, :4]
    obj_scores = predictions[:, 4:5]
    cls_scores = predictions[:, 5:]
    combined = obj_scores * cls_scores

    boxes_xyxy = np.empty_like(boxes_cxcy)
    boxes_xyxy[:, 0] = boxes_cxcy[:, 0] - boxes_cxcy[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes_cxcy[:, 1] - boxes_cxcy[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes_cxcy[:, 0] + boxes_cxcy[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes_cxcy[:, 1] + boxes_cxcy[:, 3] / 2.0
    boxes_xyxy /= ratio

    dets, _ = multiclass_nms(boxes_xyxy, combined, nms_thr=nms_thr, score_thr=score_thr)
    if dets is None:
        return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)

    # `final_cls` indexes cls_scores columns (0-based); person_class_id is an
    # absolute output class id starting at class_id_base, so translate it
    # back to a column index before comparing.
    person_col = decode_spec.person_class_id - decode_spec.class_id_base
    final_boxes = dets[:, :4]
    final_scores = dets[:, 4]
    final_cls = dets[:, 5]
    mask = (final_scores > 0.3) & (final_cls == person_col)
    return final_boxes[mask].astype(np.float64), final_scores[mask].astype(np.float64)


def yolox_detection_decode(
    raw: list[NDArray],
    ratio: float,
    model_input_size: tuple[int, int],
    score_threshold: float,
    nms_threshold: float,
    decode_spec: DetectionDecodeSpec,
) -> tuple[NDArray, NDArray]:
    """Decode raw YOLOX ONNX outputs for a single image into `(boxes, scores)`.

    Dispatches on what this specific export emits: 2 outputs means the ONNX
    graph's baked-in NMS was stripped (pre-NMS boxes + scores, handled by the
    generic `decode_prenms`); a single output tensor carries either
    NMS-baked-in rows (`shape[-1] == 5`, handled by the generic
    `decode_nms_baked_in`) or a raw anchor grid (`shape[-1] == 4`) needing
    YOLOX-specific grid decode. This top-level dispatch is a structural
    property of how many tensors this export's ONNX graph emits, not itself a
    `decode`-driven branch.
    """
    if len(raw) == 2:
        return decode_prenms(
            boxes=raw[0],
            scores=raw[1],
            ratio=ratio,
            score_thr=score_threshold,
            nms_thr=nms_threshold,
            decode_spec=decode_spec,
        )

    outputs_one = raw[0]
    if outputs_one.shape[-1] == 5:
        return decode_nms_baked_in(
            outputs_one=outputs_one,
            ratio=ratio,
            score_thr=score_threshold,
            decode_spec=decode_spec,
        )
    if outputs_one.shape[-1] == 4:
        return _decode_yolox_raw_anchor_grid(
            outputs_one=outputs_one,
            ratio=ratio,
            model_input_size=model_input_size,
            score_thr=score_threshold,
            nms_thr=nms_threshold,
            decode_spec=decode_spec,
        )
    raise RuntimeError(
        f"Unexpected YOLOX output shape: {outputs_one.shape}. Expected last dim 4 or 5."
    )
