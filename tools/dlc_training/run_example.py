"""Train and reload the official example in the isolated DLC environment."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("DLClight", "True")
os.environ.setdefault("MPLBACKEND", "Agg")


def prepare_run(source: Path, run_directory: Path) -> None:
    # Each experiment has an explicit new directory; never overwrite a prior run.
    run_directory.mkdir(parents=True, exist_ok=False)
    project = run_directory / "openfield"
    shutil.copytree(source, project)
    import deeplabcut
    import numpy as np
    from deeplabcut.core.engine import Engine

    np.random.seed(20261101)
    deeplabcut.load_demo_data(project / "config.yaml", engine=Engine.PYTORCH)
    (run_directory / "run_settings.json").write_text(
        json.dumps(
            {
                "seed": 20261101,
                "deeplabcut_version": deeplabcut.__version__,
                "purpose": "checkpoint and reference-prediction foundation, not accuracy certification",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def train_run(run_directory: Path, epochs: int, device: str) -> None:
    import deeplabcut
    from deeplabcut.core.engine import Engine

    started = time.perf_counter()
    deeplabcut.train_network(
        run_directory / "openfield/config.yaml",
        engine=Engine.PYTORCH,
        device=device,
        epochs=epochs,
        save_epochs=1,
        batch_size=4,
        pytorch_cfg_updates={
            "train_settings.dataloader_workers": 0,
            "train_settings.seed": 20261101,
        },
    )
    (run_directory / "training_summary.json").write_text(
        json.dumps(
            {
                "epochs_requested": epochs,
                "device": device,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def evaluate_run(run_directory: Path, device: str) -> None:
    import deeplabcut
    import numpy as np
    from deeplabcut.core.engine import Engine
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_model_snapshots
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    from deeplabcut.pose_estimation_pytorch.task import Task
    from deeplabcut.utils import auxiliaryfunctions

    config_path = run_directory / "openfield/config.yaml"
    started = time.perf_counter()
    cfg = auxiliaryfunctions.read_config(config_path)
    model_folder = cfg.project_path / auxiliaryfunctions.get_model_folder(
        cfg["TrainingFraction"][0], 1, cfg, engine=Engine.PYTORCH
    )
    train_folder = model_folder / "train"
    model_cfg = PoseConfig.from_any(train_folder / Engine.PYTORCH.pose_cfg_name)
    selected_snapshot = get_model_snapshots(
        -1, train_folder, Task(model_cfg["method"])
    )[0]
    deeplabcut.evaluate_network(
        config_path, engine=Engine.PYTORCH, device=device, plotting=False
    )
    images = sorted((run_directory / "openfield/labeled-data").rglob("*.png"))
    if not images:
        raise ValueError("No reference images found")
    output = run_directory / "reference_predictions"
    output.mkdir(exist_ok=False)
    predictions = deeplabcut.analyze_images(
        config_path,
        images=images,
        destfolder=output,
        device=device,
        plotting=False,
        snapshot_index=-1,
    )
    # DLC's reference API returns per-image dictionaries. Convert only at this
    # offline boundary; the portable artifact itself contains dense tensors.
    image_keys = sorted(predictions)
    keypoint_values = np.stack([predictions[key]["bodyparts"] for key in image_keys])
    if not np.isfinite(keypoint_values).all():
        raise ValueError("Reference predictions contain non-finite values")
    np.savez_compressed(
        output / "keypoint_predictions.npz", keypoint_values=keypoint_values
    )
    checkpoints = sorted((run_directory / "openfield").rglob("*.pt"))
    if not checkpoints:
        raise ValueError("Training did not produce a checkpoint")
    metadata = {
        "keypoint_names": list(cfg["bodyparts"]),
        "component_names": ["x", "y", "confidence"],
        "coordinate_space": "original image pixels",
        "shape": list(keypoint_values.shape),
        "selected_checkpoint": str(selected_snapshot.path.relative_to(run_directory)),
        "selected_checkpoint_sha256": hashlib.sha256(
            selected_snapshot.path.read_bytes()
        ).hexdigest(),
        "device": device,
        "evaluation_and_prediction_seconds": time.perf_counter() - started,
        "image_paths": [
            str(Path(key).relative_to(run_directory)) for key in image_keys
        ],
        "checkpoints": [
            {
                "path": str(path.relative_to(run_directory)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in checkpoints
        ],
    }
    (output / "prediction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train", "evaluate"])
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        prepare_run(args.source.resolve(), args.run_directory.resolve())
    elif args.action == "train":
        if args.epochs < 1:
            parser.error("epochs must be positive")
        train_run(args.run_directory.resolve(), args.epochs, args.device)
    else:
        evaluate_run(args.run_directory.resolve(), args.device)
