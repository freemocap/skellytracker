"""
ONNX graph surgery to make a YOLOX model accept a dynamic batch dimension.

The rtmlib-distributed YOLOX checkpoints are exported with a static `batch=1`
input dim, so batched inference falls back to a per-image serial loop.

The fix is in two passes:

Pass 1 — trivial static rewrites:
  • Input and output value_info: dim_value=1 → dim_param='N'
  • Reshape nodes whose target is a fully-static shape like [1,12,320,320]:
    change arr[0] from 1 to -1. Safe when there is no other -1 in the shape.

Pass 2 — dynamic shape computation for Reshapes that already contain -1:
  Targets like [1,-1,4], [1,-1,80], [1,-1], [1,3,320,2,-1,2] have a -1 in a
  non-batch slot, so we cannot just write arr[0]=-1 (two -1s → invalid Reshape).
  Instead we insert a short ONNX subgraph:
      Shape(data) → [N,D1,D2,...]
      Gather([N,...], [0])   → [N]       (1-element 1-D tensor)
      Constant([arr[1:]])    → tail dims
      Concat([N_1d, tail])   → [N,arr[1],arr[2],...]
  and patch the Reshape node to use this computed target.

NMS bypass via extracted subgraph:
  rtmlib's YOLOX checkpoints bake in a NonMaxSuppression subgraph that contains
  a Squeeze(axis=0) node. That node only works for batch=1. This module exposes
  pre-NMS Identity outputs so that a stripped pre-NMS session can be built for
  batch > 1 inference.
"""
import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

logger = logging.getLogger(__name__)

_BATCH_PARAM = "N"
_DYNBATCH_SUFFIX = ".dynbatch_v2.onnx"
_PRENMS_SUFFIX = ".prenms.onnx"

PRENMS_BBOX_OUTPUT = "_db_bbox_prenms"
PRENMS_CONF_OUTPUT = "_db_conf_prenms"


def ensure_dynamic_batch(src_path: str | Path) -> Path:
    """Return a path to a YOLOX ONNX model whose batch dim is symbolic.

    Idempotent: produces (or reuses) a sibling `<name>.dynbatch_v2.onnx` file.
    Regenerates if missing or older than the source.
    """
    src = Path(src_path)
    dst = src.parent / (src.name + _DYNBATCH_SUFFIX)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        logger.debug(f"Dynamic-batch ONNX cache hit: {dst}")
        return dst
    logger.info(f"Rewriting YOLOX ONNX for dynamic batch: {src} -> {dst}")
    return make_dynamic_batch_onnx(src, dst)


def ensure_prenms_model(dynbatch_path: str | Path) -> Path | None:
    """Return a path to a stripped ONNX containing only the YOLOX backbone+decode.

    Idempotent: produces (or reuses) a sibling `<name>.prenms.onnx` file.
    Returns None if the dynbatch ONNX does not have the expected pre-NMS outputs.
    """
    import onnx.utils

    src = Path(dynbatch_path)
    stem = src.name
    if stem.endswith(_DYNBATCH_SUFFIX):
        stem = stem[: -len(_DYNBATCH_SUFFIX)]
    dst = src.parent / (stem + _PRENMS_SUFFIX)

    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        logger.debug(f"Pre-NMS ONNX cache hit: {dst}")
        return dst

    model = onnx.load(str(src))
    output_names = {o.name for o in model.graph.output}
    if PRENMS_BBOX_OUTPUT not in output_names or PRENMS_CONF_OUTPUT not in output_names:
        logger.debug(
            "Dynbatch ONNX has no pre-NMS Identity outputs; "
            "YOLOX has no baked-in NMS to strip — skipping prenms extraction."
        )
        return None

    input_names = [inp.name for inp in model.graph.input]
    logger.info(f"Extracting pre-NMS ONNX backbone subgraph: {src} -> {dst}")
    onnx.utils.extract_model(
        str(src),
        str(dst),
        input_names,
        [PRENMS_BBOX_OUTPUT, PRENMS_CONF_OUTPUT],
        check_model=False,
    )
    logger.info(f"Wrote pre-NMS ONNX: {dst}")
    return dst


def make_dynamic_batch_onnx(src_path: str | Path, dst_path: str | Path) -> Path:
    """Rewrite `src_path` for dynamic-batch inference and save to `dst_path`."""
    src = Path(src_path)
    dst = Path(dst_path)

    model = onnx.load(str(src))
    graph = model.graph

    _symbolize_batch_dim(graph.input[0])
    for out in graph.output:
        _symbolize_batch_dim(out)

    constant_tensors = _constant_tensor_names(graph)

    pass1, pass2 = 0, 0
    new_nodes: list = []
    new_inits: list = []
    uid = [0]

    init_map = {init.name: init for init in graph.initializer}

    for node in graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        data_name = node.input[0]
        shape_name = node.input[1]

        if data_name in constant_tensors:
            continue

        init = init_map.get(shape_name)
        if init is None or init.data_type != TensorProto.INT64:
            continue

        arr = numpy_helper.to_array(init)
        if arr.ndim != 1 or arr.size < 2 or int(arr[0]) != 1:
            continue

        n_neg1 = int(np.sum(arr == -1))
        if n_neg1 == 0:
            new_arr = arr.copy()
            new_arr[0] = -1
            new_init = numpy_helper.from_array(new_arr.astype(np.int64), name=init.name)
            _replace_initializer(graph, init.name, new_init)
            init_map[init.name] = new_init
            pass1 += 1
        else:
            uid[0] += 1
            extra_nodes, extra_inits, target_output = _build_dynamic_target(
                data_name=data_name,
                arr=arr,
                uid=uid[0],
            )
            new_nodes.extend(extra_nodes)
            new_inits.extend(extra_inits)
            node.input[1] = target_output
            pass2 += 1

    for n in new_nodes:
        graph.node.append(n)
    for ini in new_inits:
        graph.initializer.append(ini)

    while graph.value_info:
        graph.value_info.pop()

    logger.debug(f"Pass-1 Reshape fixes: {pass1}  Pass-2 (dynamic) Reshape fixes: {pass2}")

    nms_bypass_added = _add_prenms_outputs(graph, constant_tensors)
    if nms_bypass_added:
        logger.info(
            f"Pre-NMS bypass outputs added ({PRENMS_BBOX_OUTPUT!r}, "
            f"{PRENMS_CONF_OUTPUT!r}). Batch>1 YOLOX inference will use Python NMS."
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    logger.info(f"Wrote dynamic-batch YOLOX ONNX: {dst}")
    return dst


def _find_prenms_tensors(graph, constant_names: set[str]) -> tuple[str | None, str | None]:
    init_names = {init.name for init in graph.initializer}

    consumers: dict[str, list] = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    conf_tensor: str | None = None
    squeeze_out: str | None = None

    for node in graph.node:
        if node.op_type != "Squeeze":
            continue
        if not node.input or node.input[0] in constant_names:
            continue

        axes: list[int] = []
        for attr in node.attribute:
            if attr.name == "axes":
                axes = list(attr.ints)
        if not axes and len(node.input) > 1 and node.input[1] in init_names:
            for init in graph.initializer:
                if init.name == node.input[1]:
                    axes = list(numpy_helper.to_array(init).flat)
                    break

        if 0 not in axes:
            continue

        cand_out = node.output[0] if node.output else None
        if cand_out is None:
            continue
        if not any(c.op_type == "TopK" for c in consumers.get(cand_out, [])):
            continue

        conf_tensor = node.input[0]
        squeeze_out = cand_out
        break

    if conf_tensor is None:
        return None, None

    topk_indices: str | None = None
    for consumer_node in consumers.get(squeeze_out, []):
        if consumer_node.op_type == "TopK" and len(consumer_node.output) >= 2:
            topk_indices = consumer_node.output[1]
            break

    if topk_indices is None:
        return None, conf_tensor

    bbox_tensor: str | None = None
    for consumer_node in consumers.get(topk_indices, []):
        if consumer_node.op_type != "Gather":
            continue
        data_input = consumer_node.input[0]
        if data_input not in constant_names:
            bbox_tensor = data_input
            break

    return bbox_tensor, conf_tensor


def _add_prenms_outputs(graph, constant_names: set[str]) -> bool:
    bbox_tensor, conf_tensor = _find_prenms_tensors(graph, constant_names)
    if conf_tensor is None:
        return False

    existing_outputs = {o.name for o in graph.output}
    added = 0

    if conf_tensor not in existing_outputs:
        graph.node.append(helper.make_node(
            "Identity", inputs=[conf_tensor], outputs=[PRENMS_CONF_OUTPUT],
        ))
        graph.output.append(helper.make_tensor_value_info(
            PRENMS_CONF_OUTPUT, TensorProto.FLOAT, None,
        ))
        added += 1

    if bbox_tensor is not None and bbox_tensor not in existing_outputs:
        graph.node.append(helper.make_node(
            "Identity", inputs=[bbox_tensor], outputs=[PRENMS_BBOX_OUTPUT],
        ))
        graph.output.append(helper.make_tensor_value_info(
            PRENMS_BBOX_OUTPUT, TensorProto.FLOAT, None,
        ))
        added += 1

    return added == 2


def _symbolize_batch_dim(value_info) -> None:
    tensor_type = value_info.type.tensor_type
    shape = tensor_type.shape
    if not shape.dim:
        return
    leading = shape.dim[0]
    if leading.HasField("dim_value") and leading.dim_value == 1:
        leading.ClearField("dim_value")
        leading.dim_param = _BATCH_PARAM
    elif not leading.HasField("dim_param"):
        leading.dim_param = _BATCH_PARAM


def _constant_tensor_names(graph) -> set[str]:
    names: set[str] = {init.name for init in graph.initializer}
    for node in graph.node:
        if node.op_type in ("Constant", "ConstantOfShape"):
            names.update(node.output)
    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if all(inp in names or inp == "" for inp in node.input):
                new = set(node.output) - names
                if new:
                    names |= new
                    changed = True
    return names


def _replace_initializer(graph, name: str, new_init) -> None:
    for i, existing in enumerate(graph.initializer):
        if existing.name == name:
            graph.initializer.remove(existing)
            graph.initializer.insert(i, new_init)
            return


def _build_dynamic_target(
        data_name: str,
        arr: np.ndarray,
        uid: int,
) -> tuple[list, list, str]:
    nodes = []
    inits = []

    shape_out = f"_db_shape_{uid}"
    nodes.append(helper.make_node("Shape", inputs=[data_name], outputs=[shape_out]))

    idx_name = f"_db_idx_{uid}"
    inits.append(numpy_helper.from_array(np.array([0], dtype=np.int64), name=idx_name))
    batch_1d = f"_db_batch_{uid}"
    nodes.append(helper.make_node(
        "Gather", inputs=[shape_out, idx_name], outputs=[batch_1d], axis=0,
    ))

    tail = arr[1:]

    if tail.size == 0:
        target_out = batch_1d
    else:
        tail_name = f"_db_tail_{uid}"
        inits.append(numpy_helper.from_array(tail.astype(np.int64), name=tail_name))
        target_out = f"_db_target_{uid}"
        nodes.append(helper.make_node(
            "Concat", inputs=[batch_1d, tail_name], outputs=[target_out], axis=0,
        ))

    return nodes, inits, target_out


def ensure_prenms_for_coreml(src_path: str | Path) -> Path:
    """Prepare a YOLOX model for CoreML by stripping the NMS subgraph.

    CoreML cannot compile the baked-in NonMaxSuppression nodes. This chains
    the dynamic-batch surgery (which adds pre-NMS bypass outputs) with
    prenms extraction (which removes everything after those outputs), giving
    CoreML a clean backbone it can compile. Falls back to the dynbatch model
    if prenms extraction is not possible (model has no baked-in NMS).
    """
    dynbatch_path = ensure_dynamic_batch(src_path)
    prenms_path = ensure_prenms_model(dynbatch_path)
    if prenms_path is None:
        logger.info(
            "No baked-in NMS found in %s; using dynbatch model for CoreML.",
            Path(src_path).name,
        )
        return dynbatch_path
    return prenms_path


__all__ = [
    "ensure_dynamic_batch",
    "ensure_prenms_for_coreml",
    "ensure_prenms_model",
    "make_dynamic_batch_onnx",
    "PRENMS_BBOX_OUTPUT",
    "PRENMS_CONF_OUTPUT",
]
