"""
ONNX graph surgery to make a YOLOX model accept a dynamic batch dimension.

The rtmlib-distributed YOLOX checkpoints are exported with a static `batch=1`
input dim, so `RTMPoseSession._detect_persons_batched` falls back to a per-image
serial loop (see `_probe_supports_batch` in `rtmpose_session.py`).

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

Pass 2 applies to all Reshape nodes whose data input is computed (not a constant
initializer/Constant-op), whose target initializer has leading 1, and where a
Pass-1 fix would produce a double -1.

NMS bypass via extracted subgraph:
  rtmlib's YOLOX checkpoints bake in a NonMaxSuppression subgraph that contains
  a Squeeze(axis=0) node. That node only works for batch=1. ORT's CUDA EP
  compiles and runs the entire graph even when only a subset of outputs is
  requested, so we cannot skip Squeeze by requesting pre-NMS outputs from the
  full session.

  The fix: use onnx.utils.extract_model to create a second, stripped ONNX
  (<name>.prenms.onnx) that contains ONLY the backbone+decode path up to the
  pre-NMS bbox and confidence tensors. This model physically has no Squeeze or
  NMS nodes, so a dedicated ORT session built from it runs purely the backbone
  and decode for any batch size. RTMPoseSession uses this second session for
  batch>1 and applies Python NMS per image.
"""
import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

logger = logging.getLogger(__name__)

_BATCH_PARAM = "N"
# Bump suffix version to force cache regeneration when graph surgery changes.
_DYNBATCH_SUFFIX = ".dynbatch_v2.onnx"
# Extracted backbone-only model (no Squeeze, no NMS) used for batch>1 detection.
_PRENMS_SUFFIX = ".prenms.onnx"

# Stable output names in the dynbatch ONNX (Identity nodes) used to locate the
# pre-NMS tensors when extracting the prenms subgraph.
PRENMS_BBOX_OUTPUT = "_db_bbox_prenms"
PRENMS_CONF_OUTPUT = "_db_conf_prenms"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    Uses onnx.utils.extract_model on the dynbatch ONNX to produce a model whose
    outputs are PRENMS_BBOX_OUTPUT and PRENMS_CONF_OUTPUT. The resulting model
    has no Squeeze or NMS nodes, so an ORT session built from it runs correctly
    for any batch size.

    Idempotent: produces (or reuses) a sibling `<name>.prenms.onnx` file.
    Returns None if the dynbatch ONNX does not have the expected pre-NMS outputs
    (i.e., the source YOLOX was exported without baked-in NMS).
    """
    import onnx.utils

    src = Path(dynbatch_path)
    # Derive the prenms path from the original model stem (strip the dynbatch suffix).
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
    # check_model=False: skip the pre-extraction shape check since we cleared
    # graph.value_info during Reshape surgery (stale batch=1 shapes dropped).
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

    # --- mark input / output batch dims as symbolic ---
    _symbolize_batch_dim(graph.input[0])
    for out in graph.output:
        _symbolize_batch_dim(out)

    # --- collect constant-producing tensor names (don't touch their Reshapes) ---
    constant_tensors = _constant_tensor_names(graph)

    # --- two-pass Reshape surgery ---
    pass1, pass2 = 0, 0
    new_nodes: list = []
    new_inits: list = []
    uid = [0]  # mutable counter shared across calls

    init_map = {init.name: init for init in graph.initializer}

    for node in graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        data_name = node.input[0]
        shape_name = node.input[1]

        # skip Reshapes whose data is constant (anchor grids etc.)
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
            # Pass 1: no existing -1 → simple leading-dim swap
            new_arr = arr.copy()
            new_arr[0] = -1
            new_init = numpy_helper.from_array(new_arr.astype(np.int64), name=init.name)
            _replace_initializer(graph, init.name, new_init)
            init_map[init.name] = new_init
            pass1 += 1
        else:
            # Pass 2: has -1 elsewhere → must compute batch dim dynamically
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

    # Insert new nodes and initializers
    for n in new_nodes:
        graph.node.append(n)
    for ini in new_inits:
        graph.initializer.append(ini)

    # Drop stale intermediate shape annotations (they encoded the old batch=1 shapes)
    while graph.value_info:
        graph.value_info.pop()

    logger.debug(f"Pass-1 Reshape fixes: {pass1}  Pass-2 (dynamic) Reshape fixes: {pass2}")

    # --- expose pre-NMS tensors so the NMS subgraph can be bypassed for N>1 ---
    nms_bypass_added = _add_prenms_outputs(graph, constant_tensors)
    if nms_bypass_added:
        logger.info(
            f"Pre-NMS bypass outputs added ({PRENMS_BBOX_OUTPUT!r}, "
            f"{PRENMS_CONF_OUTPUT!r}). Batch>1 YOLOX inference will use Python NMS."
        )
    else:
        logger.debug(
            "No Squeeze(axis=0)→TopK pattern found; YOLOX has no baked-in NMS "
            "to bypass. Standard batched output will be used."
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    logger.info(f"Wrote dynamic-batch YOLOX ONNX: {dst}")
    return dst


# ---------------------------------------------------------------------------
# Pre-NMS bypass helpers
# ---------------------------------------------------------------------------

def _find_prenms_tensors(
        graph,
        constant_names: set[str],
) -> tuple[str | None, str | None]:
    """Find pre-NMS bbox and confidence tensors in a YOLOX-with-NMS ONNX graph.

    Structural search:
      1. Squeeze(axis=0) on non-constant data whose output feeds a TopK node
         → conf_tensor is that Squeeze's data input
      2. TopK output[1] (indices) → Gather on non-constant data
         → bbox_tensor is that Gather's data input

    Returns (bbox_tensor_name, conf_tensor_name) or (None, None) if not found.
    """
    init_names = {init.name for init in graph.initializer}

    # Build tensor → consumer nodes map
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

        # Axes: opset < 13 uses attribute; opset 13+ uses second input initializer.
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

        # Confirm that this Squeeze feeds a TopK (to avoid false positives).
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

    # Find TopK consuming squeeze output.
    topk_indices: str | None = None
    for consumer_node in consumers.get(squeeze_out, []):
        if consumer_node.op_type == "TopK" and len(consumer_node.output) >= 2:
            topk_indices = consumer_node.output[1]
            break

    if topk_indices is None:
        return None, conf_tensor

    # Find Gather on non-constant data using TopK indices → pre-NMS bbox tensor.
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
    """Expose pre-NMS bbox and confidence tensors as additional graph outputs.

    For YOLOX models with NMS baked into the ONNX, Squeeze(axis=0) in the NMS
    subgraph fails for batch > 1. By adding these named outputs, RTMPoseSession
    can request them directly — ORT will execute only the backbone + decode,
    skipping the Squeeze/NMS subgraph entirely — and apply Python NMS per image.

    Returns True if both outputs were successfully added.
    """
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

    logger.debug(
        f"Pre-NMS tensors: bbox={bbox_tensor!r}, conf={conf_tensor!r}; "
        f"added {added} output(s)."
    )
    return added == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _symbolize_batch_dim(value_info) -> None:
    """Replace a hard `dim_value=1` on the leading axis with `dim_param='N'`."""
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
    """All tensor names that are definitely constant (initializers or outputs of
    Constant/ConstantOfShape nodes)."""
    names: set[str] = {init.name for init in graph.initializer}
    for node in graph.node:
        if node.op_type in ("Constant", "ConstantOfShape"):
            names.update(node.output)
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
    """Build ONNX nodes that compute a Reshape target [N, arr[1], arr[2], ...]
    where N is the dynamic batch size of `data_name`.

    Returns (new_nodes, new_initializers, output_tensor_name).
    """
    nodes = []
    inits = []

    # Shape(data) → 1-D int64 tensor [N, D1, D2, ...]
    shape_out = f"_db_shape_{uid}"
    nodes.append(helper.make_node("Shape", inputs=[data_name], outputs=[shape_out]))

    # Gather(shape_out, indices=[0], axis=0) → [N]  (1-D, 1 element)
    idx_name = f"_db_idx_{uid}"
    inits.append(numpy_helper.from_array(np.array([0], dtype=np.int64), name=idx_name))
    batch_1d = f"_db_batch_{uid}"
    nodes.append(helper.make_node(
        "Gather", inputs=[shape_out, idx_name], outputs=[batch_1d], axis=0,
    ))

    tail = arr[1:]  # everything after the batch slot

    if tail.size == 0:
        # Target is just [1] → [N]; no concat needed
        target_out = batch_1d
    else:
        tail_name = f"_db_tail_{uid}"
        inits.append(numpy_helper.from_array(tail.astype(np.int64), name=tail_name))
        target_out = f"_db_target_{uid}"
        nodes.append(helper.make_node(
            "Concat", inputs=[batch_1d, tail_name], outputs=[target_out], axis=0,
        ))

    return nodes, inits, target_out


__all__ = [
    "ensure_dynamic_batch",
    "ensure_prenms_model",
    "make_dynamic_batch_onnx",
    "PRENMS_BBOX_OUTPUT",
    "PRENMS_CONF_OUTPUT",
]
