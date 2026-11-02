"""Prepare pinned candidate graphs for existing NCHW sessions.

The raw NHWC graphs and fixed palm reshape constants are documented in
[@opencvZooHand2023] and [@opencvZooPalm2023]. Preparation runs outside capture.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
from filelock import FileLock

# Only these inspected artifacts may undergo this model-specific transformation.
CANDIDATE_HASHES = {
    "handpose_estimation_mediapipe_2023feb.onnx": "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c",
    "palm_detection_mediapipe_2023feb.onnx": "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c",
}
PREPARATION_VERSION = "nchw-batch-v2"


def prepare_candidate_model(source: Path, *, pack_outputs: bool = False) -> Path:
    """Validate source bytes and atomically prepare an independently cached graph."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    source = Path(source)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest not in CANDIDATE_HASHES.values():
        raise ValueError("Unrecognized MediaPipe ONNX candidate SHA-256")
    destination = source.with_name(
        f"{source.stem}.{digest[:12]}.{PREPARATION_VERSION}{'-packed-v1' if pack_outputs else ''}.onnx"
    )
    with FileLock(str(destination) + ".lock"):
        if destination.exists():
            onnx.checker.check_model(str(destination))
            return destination
        model = onnx.load(source)
        if len(model.graph.input) != 1:
            raise ValueError("Candidate must have exactly one image input")
        image_input = model.graph.input[0]
        shape = [
            dimension.dim_value for dimension in image_input.type.tensor_type.shape.dim
        ]
        if len(shape) != 4 or shape[0] != 1 or shape[-1] != 3:
            raise ValueError(
                f"Expected inspected NHWC image signature, received {shape}"
            )
        if image_input.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise ValueError("Candidate must accept float32 images")
        original_name = image_input.name
        internal_name = original_name + "_original_nhwc"
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                if name == original_name:
                    node.input[index] = internal_name
        model.graph.node.insert(
            0,
            helper.make_node(
                "Transpose",
                [original_name],
                [internal_name],
                perm=[0, 2, 3, 1],
                name="candidate_nchw_to_nhwc",
            ),
        )
        image_input.CopyFrom(
            helper.make_tensor_value_info(
                original_name,
                TensorProto.FLOAT,
                ["batch", shape[-1], shape[1], shape[2]],
            )
        )
        if digest == CANDIDATE_HASHES["palm_detection_mediapipe_2023feb.onnx"]:
            initializers = {value.name: value for value in model.graph.initializer}
            updated = 0
            prepared_shapes: set[str] = set()
            for node in model.graph.node:
                if node.op_type != "Reshape" or node.input[1] not in initializers:
                    continue
                updated += 1
                if node.input[1] in prepared_shapes:
                    continue
                prepared_shapes.add(node.input[1])
                tensor = initializers[node.input[1]]
                values = numpy_helper.to_array(tensor).copy()
                if values.ndim != 1 or values.tolist() not in ([1, -1, 18], [1, -1, 1]):
                    raise ValueError(f"Unexpected pinned palm reshape: {node.name}")
                values[0] = 0  # ONNX reshape copies the runtime batch dimension.
                tensor.CopyFrom(numpy_helper.from_array(values, name=tensor.name))
            if updated != 4:
                raise ValueError(
                    f"Expected four pinned palm reshape nodes, received {updated}"
                )

            # The exported resize sizes also bake in batch=1. Preserve spatial
            # dimensions and channels while reading batch from the runtime input.
            model.graph.initializer.append(
                numpy_helper.from_array(
                    np.array([0], dtype=np.int64), name="candidate_batch_index"
                )
            )
            nodes = [
                helper.make_node("Shape", [original_name], ["candidate_input_shape"]),
                helper.make_node(
                    "Gather",
                    ["candidate_input_shape", "candidate_batch_index"],
                    ["candidate_batch_size"],
                    axis=0,
                ),
            ]
            resized = 0
            for node in model.graph.node:
                if node.op_type == "Resize":
                    sizes = numpy_helper.to_array(initializers[node.input[3]])
                    if sizes.ndim != 1 or len(sizes) != 4 or sizes[0] != 1:
                        raise ValueError(f"Unexpected pinned palm resize: {node.name}")
                    tail_name = f"candidate_resize_tail_{resized}"
                    sizes_name = f"candidate_resize_sizes_{resized}"
                    model.graph.initializer.append(
                        numpy_helper.from_array(sizes[1:].copy(), name=tail_name)
                    )
                    nodes.append(
                        helper.make_node(
                            "Concat",
                            ["candidate_batch_size", tail_name],
                            [sizes_name],
                            axis=0,
                        )
                    )
                    node.input[3] = sizes_name
                    resized += 1
                nodes.append(node)
            if resized != 2:
                raise ValueError(
                    f"Expected two pinned palm resize nodes, received {resized}"
                )
            del model.graph.node[:]
            model.graph.node.extend(nodes)
        for output in model.graph.output:
            output.type.tensor_type.shape.dim[0].dim_param = "batch"
        if pack_outputs:
            _pack_candidate_outputs(model, digest)
        del model.graph.value_info[:]  # Discard obsolete static intermediate shapes.
        onnx.checker.check_model(model)
        with tempfile.NamedTemporaryFile(
            dir=source.parent, suffix=".onnx", delete=False
        ) as file:
            temporary = Path(file.name)
        try:
            onnx.save_model(model, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def prepare_detector_model(source: Path) -> Path:
    """Expose all native predictions in named dense columns for the sidecar."""
    return prepare_candidate_model(source, pack_outputs=True)


def _pack_candidate_outputs(model, digest: str) -> None:
    from onnx import TensorProto, helper, numpy_helper

    raw_names = [output.name for output in model.graph.output]
    if digest == CANDIDATE_HASHES["palm_detection_mediapipe_2023feb.onnx"]:
        candidate_count = model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
        model.graph.node.append(
            helper.make_node("Concat", raw_names, ["palm_predictions"], axis=2)
        )
        output = helper.make_tensor_value_info(
            "palm_predictions", TensorProto.FLOAT, ["batch", candidate_count, 19]
        )
    else:
        coordinate_count = model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
        point_count, remainder = divmod(coordinate_count, 3)
        if remainder:
            raise ValueError("Hand coordinates are not complete xyz triples")
        constants = {
            "hand_xyz_shape": [0, point_count, 3],
            "hand_scalar_shape": [0, 1, 1],
            "hand_scalar_tail": [point_count, 1],
            "hand_batch_index": [0],
        }
        model.graph.initializer.extend(
            numpy_helper.from_array(np.array(values, dtype=np.int64), name=name)
            for name, values in constants.items()
        )
        model.graph.node.extend(
            [
                helper.make_node(
                    "Shape", [model.graph.input[0].name], ["hand_input_shape"]
                ),
                helper.make_node(
                    "Gather",
                    ["hand_input_shape", "hand_batch_index"],
                    ["hand_batch_size"],
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["hand_batch_size", "hand_scalar_tail"],
                    ["hand_expand_shape"],
                    axis=0,
                ),
                helper.make_node(
                    "Reshape", [raw_names[0], "hand_xyz_shape"], ["image_xyz"]
                ),
                helper.make_node(
                    "Reshape", [raw_names[3], "hand_xyz_shape"], ["world_xyz"]
                ),
            ]
        )
        for raw_name, column_name in zip(
            raw_names[1:3], ("hand_presence", "handedness"), strict=True
        ):
            model.graph.node.extend(
                [
                    helper.make_node(
                        "Reshape",
                        [raw_name, "hand_scalar_shape"],
                        [column_name + "_scalar"],
                    ),
                    helper.make_node(
                        "Expand",
                        [column_name + "_scalar", "hand_expand_shape"],
                        [column_name],
                    ),
                ]
            )
        model.graph.node.append(
            helper.make_node(
                "Concat",
                ["image_xyz", "hand_presence", "handedness", "world_xyz"],
                ["hand_predictions"],
                axis=2,
            )
        )
        output = helper.make_tensor_value_info(
            "hand_predictions", TensorProto.FLOAT, ["batch", point_count, 8]
        )
    del model.graph.output[:]
    model.graph.output.append(output)
