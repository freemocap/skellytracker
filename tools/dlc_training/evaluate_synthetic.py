"""Evaluate the fixed final epoch on held-out synthetic views, using CPU only."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from synthetic_dataset import file_hash, read_dataset
from synthetic_metrics import evaluate_predictions


def write_overlays(output, arrays, indices, images, xy, errors, finite):
    overlays = output / "overlays"
    overlays.mkdir(exist_ok=True)
    failure_cases = []
    for row, (index, image_path) in enumerate(zip(indices, images, strict=True)):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for feature_index, point in enumerate(xy[row]):
            valid = bool(arrays["projection_valid"][index, feature_index])
            target = arrays["projected_xy_pixels"][index, feature_index] - 0.5
            predicted = point - 0.5
            color = "lime" if arrays["visibility"][index, feature_index] else "orange"
            if valid:
                x, y = target
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
            if np.isfinite(predicted).all():
                x, y = predicted
                draw.line((x - 3, y - 3, x + 3, y + 3), fill="magenta", width=2)
                draw.line((x - 3, y + 3, x + 3, y - 3), fill="magenta", width=2)
                if valid:
                    draw.line((*target, *predicted), fill="gray")
        valid_errors = errors[row][np.isfinite(errors[row])]
        missing = int((arrays["projection_valid"][index] & ~finite[row]).sum())
        maximum = float(valid_errors.max()) if valid_errors.size else None
        draw.text(
            (8, 8),
            f"Image {index} | mean {float(valid_errors.mean()):.1f} px | missing {missing}",
            fill="white",
            font=ImageFont.load_default(size=22),
            stroke_width=1,
            stroke_fill="black",
        )
        image.save(overlays / image_path.name)
        failure_cases.append(
            {
                "source_image_index": int(index),
                "image": str(image_path),
                "overlay": str(Path("overlays") / image_path.name),
                "nonfinite_valid_predictions": missing,
                "max_error_px": maximum,
                "feature_errors_px": [
                    float(value) if np.isfinite(value) else None
                    for value in errors[row]
                ],
            }
        )
    failure_cases.sort(
        key=lambda item: (
            item["nonfinite_valid_predictions"],
            item["max_error_px"] or -1,
        ),
        reverse=True,
    )
    review = Image.new("RGB", (960, 540), "black")
    legend = ImageDraw.Draw(review)
    legend.text(
        (12, 8),
        "Targets: green = visible, orange = occluded | Predictions: magenta crosses",
        fill="white",
        font=ImageFont.load_default(size=20),
    )
    legend.text(
        (12, 32),
        "Six largest unfiltered point errors; gray lines connect target and prediction",
        fill="white",
        font=ImageFont.load_default(size=18),
    )
    for index, failure in enumerate(failure_cases[:6]):
        with Image.open(output / failure["overlay"]) as image:
            tile = image.resize((320, 240))
            review.paste(tile, ((index % 3) * 320, (index // 3) * 240 + 60))
    review.save(output / "worst_cases_review.png")
    return failure_cases


def evaluate(run: Path, output: Path, snapshot_name: str, cache: Path):
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        DLClight="True",
        MPLBACKEND="Agg",
        HF_HOME=str(cache.resolve()),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    execution = json.loads((run / "training_execution.json").read_text())
    if execution["status"] != "completed":
        raise ValueError("Training must complete before fixed-checkpoint evaluation")
    preparation = json.loads((run / "preparation_manifest.json").read_text())
    source = Path(preparation["source"])
    for filename, key in (
        ("manifest.json", "source_manifest_sha256"),
        ("labels.npz", "source_labels_sha256"),
        ("split_membership.json", "split_membership_sha256"),
    ):
        if file_hash(source / filename) != preparation[key]:
            raise ValueError(f"Evaluation source changed since preparation: {filename}")
    _, _, arrays, groups, _, _ = read_dataset(source)
    validation_name = "development" if "development" in groups else "evaluation"
    indices = groups[validation_name]
    names = arrays["feature_names"].tolist()
    import torch
    from deeplabcut.core.engine import Engine
    from deeplabcut.pose_estimation_pytorch.apis.analyze_images import analyze_images
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_model_snapshots
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    from deeplabcut.pose_estimation_pytorch.task import Task
    from deeplabcut.utils import auxiliaryfunctions as af

    torch.set_num_threads(2)
    config_path = run / "project/config.yaml"
    config = af.read_config(config_path)
    if list(config["bodyparts"]) != names:
        raise ValueError("Trained keypoint order differs from geometry features")
    folder = (
        config.project_path
        / af.get_model_folder(
            config["TrainingFraction"][0], 1, config, engine=Engine.PYTORCH
        )
        / "train"
    )
    model_config = PoseConfig.from_any(folder / Engine.PYTORCH.pose_cfg_name)
    snapshots = get_model_snapshots("all", folder, Task(model_config["method"]))
    selected = [
        index
        for index, snapshot in enumerate(snapshots)
        if snapshot.path.name == snapshot_name
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exact final snapshot {snapshot_name}; found {[s.path.name for s in snapshots]}"
        )
    selected_index = selected[0]
    snapshot_path = snapshots[selected_index].path
    snapshot_hash = file_hash(snapshot_path)
    checkpoint = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    checkpoint_epoch = checkpoint["metadata"]["epoch"]
    del checkpoint
    if checkpoint_epoch != execution["epochs"]:
        raise ValueError("Selected checkpoint is not the fixed terminal training epoch")
    recorded = {entry["path"]: entry["sha256"] for entry in execution["checkpoints"]}
    if recorded.get(str(snapshot_path.relative_to(run))) != snapshot_hash:
        raise ValueError(
            "Selected checkpoint differs from completed training inventory"
        )
    output.mkdir(parents=True, exist_ok=False)
    images = [source / str(arrays["image_names"][index]) for index in indices]
    start = time.perf_counter()
    predictions = analyze_images(
        config_path,
        images=images,
        output_dir=output / "dlc_predictions",
        device="cpu",
        plotting=False,
        snapshot_index=selected_index,
    )
    normalized = {str(Path(key).resolve()): value for key, value in predictions.items()}
    values = np.stack([normalized[str(path.resolve())]["bodyparts"] for path in images])
    if values.shape != (len(indices), 1, len(names), 3):
        raise ValueError(f"Unexpected single-subject prediction shape {values.shape}")
    values = values[:, 0]
    xy, errors, finite, metrics = evaluate_predictions(
        values,
        arrays["projected_xy_pixels"][indices],
        arrays["projection_valid"][indices],
        arrays["visibility"][indices],
        names,
    )
    np.savez_compressed(
        output / "evaluation_predictions.npz",
        keypoint_values=values,
        geometry_edge_xy=xy,
        errors_pixels=errors,
        finite_prediction=finite,
        source_image_indices=indices,
        feature_names=arrays["feature_names"],
        physical_visibility=arrays["visibility"][indices],
        projection_valid=arrays["projection_valid"][indices],
        geometry_xy=arrays["projected_xy_pixels"][indices],
    )
    failure_cases = write_overlays(output, arrays, indices, images, xy, errors, finite)
    metadata = {
        "evaluation_split_name": validation_name,
        "snapshot": str(snapshot_path),
        "snapshot_sha256": snapshot_hash,
        "snapshot_index": selected_index,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_selection": "fixed final epoch, not validation-selected best",
        "device": "cpu",
        "torch_threads": 2,
        "source_image_indices": indices.tolist(),
        "image_paths": [str(path) for path in images],
        "feature_names": names,
        "source_labels_sha256": preparation["source_labels_sha256"],
        "elapsed_seconds": time.perf_counter() - start,
        "coordinate_conversion": "prediction edge-origin xy = DLC raster-centre xy + 0.5",
        "metrics": metrics,
        "failure_cases_ranked": failure_cases,
        "predictions_sha256": file_hash(output / "evaluation_predictions.npz"),
    }
    if file_hash(snapshot_path) != snapshot_hash:
        raise ValueError("Checkpoint changed during evaluation")
    (output / "evaluation_report.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf8"
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))


def refresh_saved_report(run: Path, output: Path):
    report_path = output / "evaluation_report.json"
    report = json.loads(report_path.read_text(encoding="utf8"))
    if file_hash(output / "evaluation_predictions.npz") != report["predictions_sha256"]:
        raise ValueError("Saved predictions changed")
    preparation = json.loads((run / "preparation_manifest.json").read_text())
    _, _, arrays, _, _, _ = read_dataset(Path(preparation["source"]))
    with np.load(output / "evaluation_predictions.npz", allow_pickle=False) as saved:
        indices = saved["source_image_indices"]
        xy, errors, finite, metrics = evaluate_predictions(
            saved["keypoint_values"],
            saved["geometry_xy"],
            saved["projection_valid"],
            saved["physical_visibility"],
            saved["feature_names"],
        )
    report["metrics"] = metrics
    report["failure_cases_ranked"] = write_overlays(
        output,
        arrays,
        indices,
        [Path(path) for path in report["image_paths"]],
        xy,
        errors,
        finite,
    )
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf8"
    )
    print(json.dumps(metrics["confidence_diagnostic"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--refresh-report", action="store_true")
    parser.add_argument("--snapshot", default="snapshot-010.pt")
    parser.add_argument("--hf-cache", type=Path, required=True)
    args = parser.parse_args()
    if args.refresh_report:
        refresh_saved_report(args.run.resolve(), args.output.resolve())
    else:
        evaluate(
            args.run.resolve(),
            args.output.resolve(),
            args.snapshot,
            args.hf_cache.resolve(),
        )
