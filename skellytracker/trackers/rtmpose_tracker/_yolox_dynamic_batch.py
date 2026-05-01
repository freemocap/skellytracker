"""
ONNX graph surgery to make a YOLOX model accept a dynamic batch dimension.

The rtmlib-distributed YOLOX checkpoints are exported with a static `batch=1`
input dim, so `RTMPoseSession._detect_persons_batched` is forced into a per-image
serial loop (see `_probe_supports_batch` in `rtmpose_session.py`). This module
rewrites the ONNX graph in-place — input/output batch axes become symbolic
('N'), and any constant `Reshape` targets that hard-code a leading 1 get a -1
in the batch slot — and caches the result on disk next to the original.

Same weights, same numerics. Just the shape declarations change so onnxruntime
will accept (N, 3, H, W) inputs with N > 1.
"""
import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

logger = logging.getLogger(__name__)

_BATCH_PARAM = "N"
_DYNBATCH_SUFFIX = ".dynbatch.onnx"


def ensure_dynamic_batch(src_path: str | Path) -> Path:
    """Return a path to a YOLOX ONNX model whose batch dim is symbolic.

    Idempotent and side-effect-free across processes: produces (or reuses) a
    sibling file `<name>.dynbatch.onnx` next to the source. If the cached copy
    is missing or older than the source, it is regenerated.
    """
    src = Path(src_path)
    dst = src.with_suffix(src.suffix + _DYNBATCH_SUFFIX) if src.suffix else (
        src.parent / (src.name + _DYNBATCH_SUFFIX)
    )
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        logger.debug(f"Dynamic-batch ONNX cache hit: {dst}")
        return dst
    logger.info(f"Rewriting YOLOX ONNX for dynamic batch: {src} -> {dst}")
    return make_dynamic_batch_onnx(src, dst)


def make_dynamic_batch_onnx(src_path: str | Path, dst_path: str | Path) -> Path:
    """Load `src_path`, rewrite batch axes to symbolic, save to `dst_path`."""
    src = Path(src_path)
    dst = Path(dst_path)

    model = onnx.load(str(src))
    graph = model.graph

    _rewrite_value_info_batch(graph.input[0])
    for output in graph.output:
        _rewrite_value_info_batch(output)

    _rewrite_reshape_batch_dims(model)

    # Drop any cached shape inferences in value_info — they encode the old batch=1.
    while len(graph.value_info) > 0:
        graph.value_info.pop()

    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    logger.info(f"Wrote dynamic-batch YOLOX ONNX: {dst}")
    return dst


def _rewrite_value_info_batch(value_info) -> None:
    """Replace a hard `dim_value=1` in the leading axis with `dim_param='N'`."""
    tensor_type = value_info.type.tensor_type
    shape = tensor_type.shape
    if not shape.dim:
        return
    leading = shape.dim[0]
    if leading.HasField("dim_value") and leading.dim_value == 1:
        leading.ClearField("dim_value")
        leading.dim_param = _BATCH_PARAM
    elif not leading.HasField("dim_param"):
        # No declared dim — set the param so downstream tools see it.
        leading.dim_param = _BATCH_PARAM


def _rewrite_reshape_batch_dims(model) -> None:
    """For every Reshape node whose `shape` initializer has a leading `1`,
    replace it with `-1`. YOLOX's exported graph contains a few of these and
    they pin the batch dim back to 1 even after we relax the input shape.
    """
    initializers = {init.name: init for init in model.graph.initializer}
    rewritten = 0
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        if len(node.input) < 2:
            continue
        shape_name = node.input[1]
        init = initializers.get(shape_name)
        if init is None:
            continue
        # Only int64 1D shape tensors are valid Reshape `shape` inputs.
        if init.data_type != TensorProto.INT64:
            continue
        arr = numpy_helper.to_array(init)
        if arr.ndim != 1 or arr.size == 0:
            continue
        if int(arr[0]) != 1:
            continue
        new_arr = arr.copy()
        new_arr[0] = -1
        # If the result has more than one -1, leave it alone — invalid Reshape.
        if int(np.sum(new_arr == -1)) > 1:
            continue
        new_init = numpy_helper.from_array(new_arr.astype(np.int64), name=init.name)
        # Replace in-place: locate by reference and update.
        for i, existing in enumerate(model.graph.initializer):
            if existing.name == init.name:
                model.graph.initializer.remove(existing)
                model.graph.initializer.insert(i, new_init)
                break
        rewritten += 1
    if rewritten:
        logger.debug(f"Relaxed batch dim in {rewritten} Reshape initializer(s)")


__all__ = ["ensure_dynamic_batch", "make_dynamic_batch_onnx"]
