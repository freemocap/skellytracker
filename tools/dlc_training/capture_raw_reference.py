"""Capture DLC-only CPU reference tensors for an independent inference importer.

This script belongs exclusively to the optional training environment. It is not
imported by conversion or tracking. APIs: [@dlc302PoseModel; @dlc302Preprocessor].
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("DLClight", "True")
os.environ.setdefault("MPLBACKEND", "Agg")


def capture_reference(run_directory: Path, output_directory: Path) -> None:
    print("Importing CPU reference dependencies", flush=True)
    import cv2
    import numpy as np
    import torch
    import yaml
    from deeplabcut.pose_estimation_pytorch.data.preprocessor import (
        build_bottom_up_preprocessor,
    )
    from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms
    from deeplabcut.pose_estimation_pytorch.models.model import PoseModel

    torch.set_num_threads(4)
    print("Loading checkpoint and configuration", flush=True)
    metadata = json.loads(
        (run_directory / "reference_predictions/prediction_metadata.json").read_text()
    )
    checkpoint = run_directory / metadata["selected_checkpoint"]
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != metadata["selected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint hash mismatch")
    config_path = checkpoint.parent / "pytorch_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    model = PoseModel.build(config["model"], pretrained_backbone=False).cpu().eval()
    print("Built official CPU model", flush=True)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["model"],
        strict=True,
    )
    preprocessor = build_bottom_up_preprocessor(
        "RGB", build_transforms(config["data"]["inference"])
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    selected_images = [run_directory / name for name in metadata["image_paths"][:3]]
    rgb_images = [
        cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        for path in selected_images
    ]
    cases = {
        "full_single": [rgb_images[0]],
        "full_pair": rgb_images[:2],
        "full_tail": rgb_images,
        "offset_crop": [rgb_images[0][37:264, 53:444]],
    }
    report = {
        "checkpoint_sha256": checkpoint_hash,
        "torch_version": torch.__version__,
        "device": "cpu",
        "keypoint_names": metadata["keypoint_names"],
        "cases": {},
        "coordinate_space": "input image or crop pixels; apply recorded inverse transform once",
        "batch_scope": "direct model batches, not shared-session padded-tail validation",
    }
    with torch.inference_mode():
        for case_name, images in cases.items():
            print(f"Capturing {case_name}", flush=True)
            normalized_inputs = torch.cat(
                [preprocessor(image, {})[0] for image in images], dim=0
            ).cpu()
            raw_outputs = model(normalized_inputs)
            poses = model.get_predictions(raw_outputs)["bodypart"]["poses"]
            artifact = output_directory / f"{case_name}.npz"
            np.savez_compressed(
                artifact,
                rgb_images=np.stack(images),
                normalized_inputs=normalized_inputs.numpy(),
                heatmaps=raw_outputs["bodypart"]["heatmap"].numpy(),
                location_refinements=raw_outputs["bodypart"]["locref"].numpy(),
                keypoint_values=poses.numpy(),
            )
            report["cases"][case_name] = {
                "input_shape": list(normalized_inputs.shape),
                "heatmap_shape": list(raw_outputs["bodypart"]["heatmap"].shape),
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "source_images": [
                    {
                        "path": str(path.relative_to(run_directory)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in selected_images[: len(images)]
                ],
                "crop_xyxy": [53, 37, 444, 264] if case_name == "offset_crop" else None,
                "inverse_transform": (
                    [[1.0, 0.0, 53.0], [0.0, 1.0, 37.0], [0.0, 0.0, 1.0]]
                    if case_name == "offset_crop"
                    else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                ),
            }
    (output_directory / "reference_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    import faulthandler

    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        capture_reference(args.run_directory.resolve(), args.output_directory.resolve())
    finally:
        faulthandler.cancel_dump_traceback_later()
