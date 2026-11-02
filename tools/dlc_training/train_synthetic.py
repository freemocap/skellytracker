"""Prepare an exact synthetic split, or explicitly train the prepared DLC project."""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from synthetic_dataset import file_hash, read_dataset, training_selection

os.environ.setdefault("DLClight", "True")
os.environ.setdefault("MPLBACKEND", "Agg")


def prepare(
    source: Path,
    output: Path,
    *,
    epochs: int = 200,
    save_epochs: int = 10,
    max_snapshots: int = 5,
    batch_size: int = 8,
    augmentation_policy: str = "dlc-default",
):
    if min(epochs, save_epochs, max_snapshots, batch_size) < 1:
        raise ValueError(
            "Epochs, save interval and checkpoint retention must be positive"
        )
    # Only this command sets CPU isolation; train selects its device explicitly.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    manifest, _membership, arrays, groups, coordinates, labeled = read_dataset(source)
    import deeplabcut
    import numpy as np
    import pandas as pd  # DLC's HDF contract requires a pandas MultiIndex table.
    from deeplabcut.core.config.utils import read_config_as_dict, write_config
    from deeplabcut.core.engine import Engine
    from deeplabcut.utils import auxiliaryfunctions as af

    output.mkdir(parents=True, exist_ok=False)
    project = output / "project"
    labels_directory = project / "labeled-data" / "synthetic"
    labels_directory.mkdir(parents=True)
    validation_name, selected = training_selection(groups)
    inverse = np.full(len(arrays["image_names"]), -1, dtype=np.int64)
    inverse[selected] = np.arange(len(selected))
    image_names = [f"source_{int(index):06d}.png" for index in selected]
    for index, name in zip(selected, image_names, strict=True):
        shutil.copyfile(
            source / str(arrays["image_names"][index]), labels_directory / name
        )
    scorer = "synthetic_geometry"
    columns = pd.MultiIndex.from_product(
        [[scorer], arrays["feature_names"].tolist(), ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    index = pd.MultiIndex.from_tuples(
        [("labeled-data", "synthetic", name) for name in image_names]
    )
    table = pd.DataFrame(
        coordinates[selected].reshape(len(selected), -1), index=index, columns=columns
    )
    table.to_hdf(
        labels_directory / f"CollectedData_{scorer}.h5", key="df_with_missing", mode="w"
    )
    config, _ = af.create_config_template()
    width, height = manifest["image_size"]
    fraction = round(len(groups["training"]) / len(selected), 2)
    config.update(
        Task="synthetic_benchy",
        scorer=scorer,
        date="Nov01",
        project_path=str(project.resolve()),
        video_sets={
            str(project / "videos" / "synthetic.mp4"): {
                "crop": f"0, {width}, 0, {height}"
            }
        },
        bodyparts=arrays["feature_names"].tolist(),
        skeleton=[],
        multianimalproject=False,
        default_net_type="resnet_50",
        default_augmenter="default",
        TrainingFraction=[fraction],
        iteration=0,
        snapshotindex=-1,
        cropping=False,
        start=0,
        stop=1,
        numframes2pick=len(selected),
        x1=0,
        x2=width,
        y1=0,
        y2=height,
        batch_size=batch_size,
        engine="pytorch",
    )
    config_path = project / "config.yaml"
    af.write_config(config_path, config)
    train_indices = inverse[groups["training"]].tolist()
    test_indices = inverse[groups[validation_name]].tolist()
    deeplabcut.create_training_dataset(
        config_path,
        Shuffles=[1],
        trainIndices=[train_indices],
        testIndices=[test_indices],
        net_type="resnet_50",
        engine=Engine.PYTORCH,
        userfeedback=False,
    )
    pose_paths = list(project.glob("dlc-models-pytorch/**/train/pytorch_config.yaml"))
    if len(pose_paths) != 1:
        raise ValueError("Expected exactly one prepared training configuration")
    # DLC's YAML IO preserves scientific-notation rates as numbers. A PyYAML
    # roundtrip can quote 1e-05 and fail at the later milestone. [@dlcConfigYamlIO302]
    pose = read_config_as_dict(pose_paths[0])
    if pose["model"]["backbone"]["model_name"] != "resnet50_gn":
        raise ValueError(
            "Prepared backbone is outside the supported conversion contract"
        )
    from training_policy import configure_training_policy

    configure_training_policy(pose, augmentation_policy)
    pose["train_settings"].update(
        batch_size=batch_size, dataloader_workers=0, seed=20261101, epochs=epochs
    )
    pose["runner"]["snapshots"].update(
        save_epochs=save_epochs, max_snapshots=max_snapshots
    )
    write_config(pose_paths[0], pose)
    # DLC rewrites its live config when training applies overrides; retain input bytes.
    shutil.copyfile(config_path, output / "prepared_project_config.yaml")
    shutil.copyfile(pose_paths[0], output / "prepared_pose_config.yaml")
    from deeplabcut.pose_estimation_pytorch.data.dlcloader import DLCLoader

    actual_split = DLCLoader.load_split(af.read_config(config_path), shuffle=1)
    if actual_split != {"train": train_indices, "test": test_indices}:
        raise ValueError(
            "DLC persisted split differs from the authoritative geometry split"
        )
    # Preserve the geometry's physical visibility, projection validity and coordinates untouched.
    for name in ("labels.npz", "manifest.json", "split_membership.json"):
        shutil.copyfile(source / name, output / ("geometry_" + name))
    np.savez_compressed(
        output / "dlc_label_boundary.npz",
        dlc_xy=coordinates,
        dlc_labeled=labeled,
        selected_source_indices=selected,
        train_row_indices=train_indices,
        evaluation_row_indices=test_indices,
    )
    roundtrip = pd.read_hdf(labels_directory / f"CollectedData_{scorer}.h5")
    np.testing.assert_array_equal(
        roundtrip.to_numpy().reshape(len(selected), len(arrays["feature_names"]), 2),
        coordinates[selected],
    )
    summary = {
        "source": str(source.resolve()),
        "source_manifest_sha256": file_hash(source / "manifest.json"),
        "source_labels_sha256": file_hash(source / "labels.npz"),
        "split_membership_sha256": file_hash(source / "split_membership.json"),
        "feature_names": arrays["feature_names"].tolist(),
        "selected_source_indices": selected.tolist(),
        "training_source_indices": groups["training"].tolist(),
        "evaluation_source_indices": groups[validation_name].tolist(),
        "evaluation_split_name": validation_name,
        "excluded_source_indices": groups.get(
            "excluded", np.empty(0, dtype=np.int64)
        ).tolist(),
        "reserved_final_source_indices": groups.get(
            "final_evaluation", np.empty(0, dtype=np.int64)
        ).tolist(),
        "dlc_image_names": image_names,
        "train_row_indices": train_indices,
        "evaluation_row_indices": test_indices,
        "coordinate_conversion": "DLC xy = geometry edge-origin xy - 0.5; invalid labels NaN only in DLC boundary",
        "training_occluded_labels_retained": int(
            (
                ~arrays["visibility"][groups["training"]] & labeled[groups["training"]]
            ).sum()
        ),
        "additional_valid_geometry_labels_masked": int(
            (arrays["projection_valid"] & ~labeled).sum()
        ),
        "pose_config": str(pose_paths[0].resolve()),
        "pose_config_sha256": file_hash(pose_paths[0]),
        "project_config_sha256": file_hash(config_path),
        "dlc_persisted_split_verified": True,
        "image_grouping": "video_sets synthetic.mp4 is a DLC labeled-image group key, not an actual video; preparation opens PNG files only",
        "deeplabcut_version": deeplabcut.__version__,
        "training_started": False,
        "augmentation_policy": augmentation_policy,
        "training_augmentation": pose["data"]["train"],
        "inference_preprocessing": pose["data"]["inference"],
        "training_batch_size": batch_size,
    }
    (output / "preparation_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf8"
    )
    print(json.dumps(summary, indent=2))


def validate_training_run(output: Path, epochs: int, device: str):
    if epochs < 1 or (
        device != "cpu" and not (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ValueError("Require positive epochs and explicit cpu or cuda:N device")
    report_path = output / "training_execution.json"
    if report_path.exists() or any((output / "project").rglob("snapshot-*.pt")):
        raise ValueError(
            "Training already started in this run; prepare a new explicit run directory"
        )
    preparation = json.loads((output / "preparation_manifest.json").read_text())
    config_path = output / "project/config.yaml"
    pose_path = Path(preparation["pose_config"])
    if (
        file_hash(config_path) != preparation["project_config_sha256"]
        or file_hash(pose_path) != preparation["pose_config_sha256"]
    ):
        raise ValueError(
            "Prepared project or pose configuration changed before training"
        )
    source = Path(preparation["source"])
    for filename, key in (
        ("manifest.json", "source_manifest_sha256"),
        ("labels.npz", "source_labels_sha256"),
        ("split_membership.json", "split_membership_sha256"),
    ):
        if file_hash(source / filename) != preparation[key]:
            raise ValueError(f"Source changed before training: {filename}")
    read_dataset(source)  # Includes exact image hashes and split validation.
    return config_path


def initialize_cuda_statistics(cuda, device: str):
    """Initialize the selected context before accessing its allocator statistics."""
    cuda.set_device(device)
    cuda.init()
    cuda.reset_peak_memory_stats(device)


def train(output: Path, epochs: int, device: str):
    config_path = validate_training_run(output, epochs, device)
    from deeplabcut.core.config.utils import read_config_as_dict

    preparation = json.loads((output / "preparation_manifest.json").read_text())
    pose_config = read_config_as_dict(preparation["pose_config"])
    if pose_config["train_settings"]["epochs"] != epochs:
        raise ValueError(
            "Requested epochs differ from the prepared training configuration"
        )
    save_epochs = pose_config["runner"]["snapshots"]["save_epochs"]
    import deeplabcut
    import torch
    from deeplabcut.core.engine import Engine

    report_path = output / "training_execution.json"
    if device.startswith("cuda:"):
        index = int(device.split(":")[1])
        if (
            not torch.cuda.is_available()
            or index < 0
            or index >= torch.cuda.device_count()
        ):
            raise ValueError(f"Requested training device is unavailable: {device}")
    if device.startswith("cuda:"):
        initialize_cuda_statistics(torch.cuda, device)
    report = {
        "status": "running",
        "process_id": os.getpid(),
        "device_name": torch.cuda.get_device_name(device)
        if device.startswith("cuda:")
        else "CPU",
        "device": device,
        "epochs": epochs,
        "batch_size": 4,
        "seed": 20261101,
        "save_epochs": save_epochs,
        "max_snapshots": pose_config["runner"]["snapshots"]["max_snapshots"],
        "preparation_manifest_sha256": file_hash(output / "preparation_manifest.json"),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf8")
    started = time.perf_counter()
    try:
        deeplabcut.train_network(
            config_path,
            engine=Engine.PYTORCH,
            device=device,
            epochs=epochs,
            save_epochs=save_epochs,
            batch_size=4,
            pytorch_cfg_updates={
                "train_settings.dataloader_workers": 0,
                "train_settings.seed": 20261101,
            },
        )
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        if device.startswith("cuda:"):
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["checkpoints"] = [
            {"path": str(path.relative_to(output)), "sha256": file_hash(path)}
            for path in sorted((output / "project").rglob("snapshot-*.pt"))
        ]
        report_path.write_text(json.dumps(report, indent=2), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify", "train"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--augmentation-policy",
        choices=["dlc-default", "diagnostic-none"],
        default="dlc-default",
    )
    parser.add_argument(
        "--save-epochs",
        type=int,
        default=10,
        help="Preparation only: checkpoint save interval",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=5,
        help="Preparation only: retained regular checkpoints",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.hf_cache.is_dir():
        parser.error("--hf-cache must name the existing local Hugging Face cache")
    os.environ["HF_HOME"] = str(args.hf_cache.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.action == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        prepare(
            args.source.resolve(),
            args.output.resolve(),
            epochs=args.epochs,
            save_epochs=args.save_epochs,
            max_snapshots=args.max_snapshots,
            batch_size=args.batch_size,
            augmentation_policy=args.augmentation_policy,
        )
    elif args.action == "verify":
        print(validate_training_run(args.output.resolve(), args.epochs, args.device))
    else:
        train(args.output.resolve(), args.epochs, args.device)
