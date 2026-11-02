"""Export an independently reconstructed model and self-describing sidecar."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import onnx
import torch
import yaml
from model_reconstruction import load_model


def export_model(
    config_path: Path,
    checkpoint_path: Path,
    output_directory: Path,
    height: int,
    width: int,
    *,
    resize_method: str = "none",
    project_config_path: Path | None = None,
) -> None:
    if height < 32 or width < 32:
        raise ValueError("Image dimensions must be at least 32 pixels")
    torch.set_num_threads(4)
    model = load_model(config_path, checkpoint_path)
    if resize_method not in ("none", "fit"):
        raise ValueError("DLC export supports native images or explicit aspect-preserving fit")
    point_names = list(model.contract.keypoint_names)
    connections = []
    if project_config_path is not None:
        project = yaml.safe_load(project_config_path.read_text(encoding="utf8"))
        if project.get("multianimalproject", False) or project.get("bodyparts") != point_names:
            raise ValueError("DLC project keypoint identities must match the supported single-subject checkpoint")
        edges = project.get("skeleton") or []
        if any(len(edge) != 2 or any(name not in point_names for name in edge) for edge in edges):
            raise ValueError("DLC project connections reference unknown keypoint identities")
        if edges:
            connections = [{"name": "dlc_connections", "edges": edges}]
    output_directory.mkdir(parents=True, exist_ok=False)
    onnx_path = output_directory / "pose_model.onnx"
    example = torch.zeros((2, 3, height, width), dtype=torch.float32)
    torch.onnx.export(
        model,
        (example,),
        onnx_path,
        input_names=["images"],
        output_names=["keypoints"],
        dynamic_shapes=({0: torch.export.Dim("batch_count", min=1, max=32)},),
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    onnx.checker.check_model(str(onnx_path), full_check=True)
    artifact_hash = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    sidecar = {
        "schema_version": "v2024.09.1019",
        "model_id": "dlc_resnet50_gn",
        "display_name": "DLC ResNet50 GroupNorm",
        "role": ["pose_estimator"],
        "input": {
            "name": "images",
            "dtype": {"fp32": "float32"},
            "layout": "NCHW",
            "normalization": "imagenet_rgb",
            "resize": ({"method": "none"} if resize_method == "none" else
                       {"method": "fit", "target_size": [height, width], "pad_value": 0}),
        },
        "outputs": [
            {
                "name": "keypoints",
                "dtype": {"fp32": "float32"},
                "shape": [-1, len(point_names), 3],
                "semantic": "keypoints",
                "fields": ["x", "y", "confidence"],
                "keypoint_axis": 1,
                "keypoint_count": len(point_names),
            }
        ],
        "pose": {
            "estimator_type": "bottom_up",
            "requires_object_detector": False,
            "tracked_points": point_names,
            "connections": connections,
            "decode": {"method": "coordinate", "coordinate_scale": "pixel"},
        },
        "sizes": {
            "native": {
                "input": {"shape": [-1, 3, height, width]},
                "onnx": {
                    "batch_artifacts": {
                        "dynamic": {
                            "precision_artifacts": {
                                "fp32": {
                                    "filename": onnx_path.name,
                                    "url_sha256": artifact_hash,
                                    "cuda_use_tf32": False,
                                }
                            }
                        }
                    }
                },
            }
        },
    }
    (output_directory / "dlc_resnet50_gn.yaml").write_text(
        yaml.safe_dump(sidecar, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "onnx_sha256": artifact_hash,
        "torch_version": torch.__version__,
        "input_size": [height, width],
        "batch_range": [1, 32],
        "keypoint_names": point_names,
        "spatial_policy": resize_method,
        "connections": connections,
        "project_config_sha256": hashlib.sha256(project_config_path.read_bytes()).hexdigest() if project_config_path else None,
        "cuda_precision_requirement": {"use_tf32": False, "input_dtype": "float32"},
    }
    (output_directory / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--resize-method", choices=("none", "fit"), default="none")
    parser.add_argument("--project-config", type=Path)
    args = parser.parse_args()
    export_model(args.config, args.checkpoint, args.output, args.height, args.width,
                 resize_method=args.resize_method, project_config_path=args.project_config)
