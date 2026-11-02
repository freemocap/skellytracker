"""Controlled native-image DLC training comparison; optional training environment only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
from functools import partial

import numpy as np
import torch

from pose_objective import linearized_pose_objective, pose_objective


def select_objective(mode, damping):
    """Select once; nonlinear retains its existing numerical settings exactly."""
    if mode == "nonlinear":
        return pose_objective
    if mode == "linearized":
        if not np.isfinite(damping) or damping <= 0:
            raise ValueError("Linearized pose damping must be finite and positive")
        return partial(linearized_pose_objective, damping=damping)
    raise ValueError(f"Unknown pose objective: {mode}")


def gradient_comparison(original, auxiliary, parameters):
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    first = torch.autograd.grad(original, parameters, retain_graph=True, allow_unused=True)
    second = torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True)
    zero = original.new_zeros(())
    first_squared = sum((value.square().sum() for value in first if value is not None), zero)
    second_squared = sum((value.square().sum() for value in second if value is not None), zero)
    product = sum(((a * b).sum() for a, b in zip(first, second) if a is not None and b is not None), zero)
    denominator = (first_squared * second_squared).sqrt()
    return {"original_gradient_l2": first_squared.sqrt().item(),
            "weighted_auxiliary_gradient_l2": second_squared.sqrt().item(),
            "gradient_cosine": (product / denominator).item() if denominator.item() > 0 else None}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode_native_offsets(outputs, stride, locref_std):
    """Native argmax coordinates; gradients flow only through selected offsets.

    Peak selection has no derivative. This is a piecewise differentiable
    diagnostic, not a straight-through estimator or a soft heatmap decoder.
    Returns DLC pixel coordinates; the geometry boundary adds 0.5 separately.
    """
    heatmaps = outputs["heatmap"]
    batch, count, height, width = heatmaps.shape
    offsets = outputs["locref"].reshape(batch, count, 2, height * width)
    indices = heatmaps.flatten(-2).argmax(-1)
    selected = offsets.gather(-1, indices[:, :, None, None].expand(-1, -1, 2, 1)).squeeze(-1)
    grid = torch.stack((indices % width, indices // width), dim=-1).to(selected.dtype)
    return (grid + 0.5) * stride + selected * locref_std


def decode_coordinates(outputs, stride, locref_std, temperature=0.025, *, decoder="spatial-expectation"):
    """Select an explicit experiment decoder; both return native DLC pixels."""
    if decoder == "native-offset":
        return decode_native_offsets(outputs, stride, locref_std)
    if decoder != "spatial-expectation":
        raise ValueError(f"Unknown coordinate decoder: {decoder}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    heatmaps = outputs["heatmap"]
    batch, count, height, width = heatmaps.shape
    offsets = outputs["locref"].reshape(batch, count, 2, height, width)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=heatmaps.device, dtype=heatmaps.dtype),
        torch.arange(width, device=heatmaps.device, dtype=heatmaps.dtype), indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=0).add(0.5).mul(stride)
    probabilities = (heatmaps.flatten(-2) / temperature).softmax(-1)
    coordinates = grid + offsets * locref_std
    return (coordinates.flatten(-2) * probabilities.unsqueeze(2)).sum(-1)


def atomic_checkpoint(path, model, optimizer, metadata):
    temporary = path.with_suffix(".tmp")
    numpy_state = np.random.get_state()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "metadata": metadata, "rng": {"torch": torch.get_rng_state(),
                "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                "python": random.getstate(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}}, temporary)
    os.replace(temporary, path)


def training_geometry(dataset, geometry, feature_names):
    """Verify training copies only; never open reserved evaluation images."""
    manifest = json.loads((geometry / "manifest.json").read_text())
    membership = json.loads((geometry / "split_membership.json").read_text())
    digest = file_hash(geometry / "labels.npz")
    if digest != manifest["label_sha256"] or digest != membership["labels_sha256"]:
        raise ValueError("Geometry label hash mismatch")
    if manifest["pixel_convention"] != "origin at top-left image edge; pixel centres at half-integers":
        raise ValueError("Unsupported geometry pixel convention")
    with np.load(geometry / "labels.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if arrays["feature_names"].tolist() != feature_names:
        raise ValueError("Geometry and model keypoint order differ")
    allowed = set(membership["splits"]["training"]["image_indices"])
    lookup = np.full(max(image["id"] for image in dataset.images) + 1, -1, dtype=np.int64)
    for image in dataset.images:
        copied = Path(image["file_name"])
        if not copied.stem.startswith("source_"):
            raise ValueError("Expected explicitly indexed source image")
        index = int(copied.stem.removeprefix("source_"))
        if index not in allowed:
            raise ValueError("Dataset contains non-training image")
        original_name = str(arrays["image_names"][index])
        expected = manifest["image_sha256"][original_name]
        if file_hash(copied) != expected:
            raise ValueError(f"Copied training image hash mismatch: {copied}")
        lookup[image["id"]] = index
    if len(dataset.images) != len(allowed) or set(lookup[lookup >= 0]) != allowed:
        raise ValueError("Training membership mismatch")
    return arrays, lookup


def run(args):
    from deeplabcut.core.config.utils import read_config_as_dict
    from torch.utils.data import DataLoader
    from deeplabcut.pose_estimation_pytorch.data.dlcloader import DLCLoader
    from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms
    from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
    from deeplabcut.pose_estimation_pytorch.task import Task

    if args.steps <= 0 or args.batch_size <= 0 or args.pose_weight < 0 or args.save_every <= 0 or args.temperature <= 0 or args.gradient_diagnostics < 0:
        raise ValueError("Positive steps/batch size and nonnegative pose weight required")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config = read_config_as_dict(args.pose_config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PoseModel.build(copy.deepcopy(config["model"]), pretrained_backbone=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).train()
    # DLC's AdamW uses model.parameters in their declared order; state load restores
    # the exact learning rate, moments and group options from the shared checkpoint.
    groups = checkpoint["optimizer"]["param_groups"]
    if len(groups) != 1 or len(groups[0]["params"]) != len(list(model.parameters())):
        raise ValueError("Unsupported optimizer parameter grouping")
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer"])
    loader = DLCLoader(args.project, trainset_index=0, shuffle=1)
    dataset = loader.create_dataset(transform=build_transforms({"normalize_images": True}),
                                    mode="train", task=Task.BOTTOM_UP)
    arrays, source_lookup = training_geometry(dataset, args.geometry, config["metadata"]["bodyparts"])
    batches = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                         generator=torch.Generator().manual_seed(args.seed))
    camera_pose = torch.as_tensor(np.linalg.inv(arrays["camera_to_object"]), dtype=torch.float32, device=device)
    intrinsics = torch.as_tensor(arrays["camera_intrinsics"], dtype=torch.float32, device=device)
    object_points = torch.as_tensor(arrays["feature_positions_mm"], dtype=torch.float32, device=device)
    projection_valid = torch.as_tensor(arrays["projection_valid"], device=device)
    predictor = config["model"]["heads"]["bodypart"]["predictor"]
    selected_objective = select_objective(args.pose_objective, args.pose_damping)
    source_files = (Path(__file__), Path(__file__).with_name("pose_objective.py"))
    source_directory = args.output / "source_snapshot"
    source_directory.mkdir()
    for source_file in source_files:
        (source_directory / source_file.name).write_bytes(source_file.read_bytes())
    manifest = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "checkpoint_sha256": file_hash(args.checkpoint), "pose_config_sha256": file_hash(args.pose_config),
                "source_sha256": file_hash(Path(__file__)),
                "pose_objective_sha256": file_hash(Path(__file__).with_name("pose_objective.py")),
                "geometry_labels_sha256": file_hash(args.geometry / "labels.npz"),
                "geometry_manifest_sha256": file_hash(args.geometry / "manifest.json"),
                "split_membership_sha256": file_hash(args.geometry / "split_membership.json"),
                "image_transform": "identity; ImageNet normalization only", "image_size": [480, 640],
                "clip_max_norm": 10.0, "scheduler_updates": False,
                "timing_policy": "diagnostic updates include separately reported gradient diagnostic overhead; not benchmark comparable",
                "optimizer_initialization": "exact shared checkpoint state", "status": "running"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not args.final_checkpoint_only:
        atomic_checkpoint(args.output / "snapshot-first.pt", model, optimizer, {"step": 0, **manifest})
    step = 0
    iterator = iter(batches)
    start = time.perf_counter()
    try:
        with (args.output / "updates.jsonl").open("w", buffering=1) as log:
            while step < args.steps:
                if (args.output / "pause.request").exists():
                    manifest["status"] = "paused"
                    break
                tick = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(batches)
                    batch = next(iterator)
                source_indices = source_lookup[batch["image_id"].numpy()]
                indices = torch.as_tensor(source_indices, device=device)
                images = batch["image"].to(device)
                if images.shape[-2:] != (480, 640):
                    raise ValueError("Native experiment requires 480x640 images")
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                targets = model.get_target(outputs, batch["annotations"])
                original = model.get_loss(outputs, targets)["total_loss"]
                xy = decode_coordinates(outputs["bodypart"], model.get_stride("bodypart"),
                                        predictor["locref_std"], args.temperature,
                                        decoder=args.coordinate_decoder) + 0.5
                labels = batch["annotations"]["keypoints"][:, 0].to(device)
                valid = projection_valid[indices] & torch.isfinite(labels[..., :2]).all(-1) & (labels[..., 2] > 0)
                result = selected_objective(xy, object_points, intrinsics[indices],
                                        camera_pose[indices, :3, :3], camera_pose[indices, :3, 3], valid)
                weighted = result.loss * (args.pose_weight if args.arm == "pose" else 0.0)
                loss = original + weighted if args.arm == 'pose' else original
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                diagnostics = {}
                diagnostic_seconds = 0.0
                if step < args.gradient_diagnostics:
                    diagnostic_start = time.perf_counter()
                    diagnostics = gradient_comparison(original, weighted, model.parameters())
                    diagnostic_seconds = time.perf_counter() - diagnostic_start
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                record = {"step": step, "original_loss": original.item(), "pose_loss": result.loss.item(),
                          "weighted_pose_loss": weighted.item(), "total_loss": loss.item(), "gradient_norm": norm.item(),
                          "valid_samples": result.valid_samples.sum().item(),
                          "failure_counts": {key: value.item() for key, value in result.failure_counts.items()},
                          "source_indices": source_indices.tolist(), "seconds": time.perf_counter() - tick}
                record.update(diagnostics, gradient_diagnostic_seconds=diagnostic_seconds)
                log.write(json.dumps(record) + "\n")
                if not args.final_checkpoint_only and step % args.save_every == 0:
                    atomic_checkpoint(args.output / f"snapshot-{step:06d}.pt", model, optimizer, {"step": step, **manifest})
            if step == args.steps:
                manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        manifest.update(completed_steps=step, elapsed_seconds=time.perf_counter() - start)
        atomic_checkpoint(args.output / "snapshot-last.pt", model, optimizer, {"step": step, **manifest})
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "pose-config", "checkpoint", "geometry", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    result.add_argument("--steps", type=int, required=True)
    result.add_argument("--arm", choices=("baseline", "pose"), required=True)
    result.add_argument("--seed", type=int, default=20261102)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--pose-weight", type=float, required=True)
    result.add_argument("--pose-objective", choices=("nonlinear", "linearized"), required=True)
    result.add_argument("--coordinate-decoder", choices=("spatial-expectation", "native-offset"), required=True,
                        help="Native-offset has gradients through selected locref only, not argmax peak selection")
    result.add_argument("--pose-damping", type=float, default=0.01,
                        help="Ridge damping for linearized objective only; nonlinear settings unchanged")
    result.add_argument("--temperature", type=float, required=True)
    result.add_argument("--save-every", type=int, default=25)
    result.add_argument("--final-checkpoint-only", action="store_true",
                        help="Save one checkpoint on completion or pause; the epoch driver retains every epoch")
    result.add_argument("--gradient-diagnostics", type=int, default=0,
                        help="First N updates: separate gradient norms/cosine; adds diagnostic overhead")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
