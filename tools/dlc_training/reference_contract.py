"""Validate offline DLC reference artifacts without importing DLC or Torch."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def validate_predictions(values: np.ndarray, metadata: dict) -> None:
    if values.dtype != np.float32 or values.ndim != 4:
        raise ValueError(
            "Expected float32 [image, subject, keypoint, component] tensor"
        )
    for field in ("keypoint_names", "component_names", "image_paths"):
        names = metadata[field]
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"{field} requires nonempty names")
        if len(names) != len(set(names)):
            raise ValueError(f"{field} contains duplicate names")
    expected = (
        len(metadata["image_paths"]),
        values.shape[1],
        len(metadata["keypoint_names"]),
        len(metadata["component_names"]),
    )
    if (
        values.shape != expected
        or list(values.shape) != metadata["shape"]
        or values.shape[1] < 1
    ):
        raise ValueError("Tensor axes disagree with artifact metadata")
    if metadata["component_names"] != ["x", "y", "confidence"]:
        raise ValueError("This DLC image-reference contract requires x, y, confidence")
    if metadata["coordinate_space"] != "original image pixels":
        raise ValueError("Reference coordinates must be original image pixels")
    if not np.isfinite(values).all():
        raise ValueError("Reference predictions must be finite")
    confidence = values[..., metadata["component_names"].index("confidence")]
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("DLC reference confidence must be in [0, 1]")


def validate_run(run_directory: Path) -> dict:
    run_directory = run_directory.resolve()
    output = run_directory / "reference_predictions"
    metadata = json.loads(
        (output / "prediction_metadata.json").read_text(encoding="utf-8")
    )
    with np.load(output / "keypoint_predictions.npz", allow_pickle=False) as archive:
        validate_predictions(archive["keypoint_values"], metadata)
    checkpoint = (run_directory / metadata["selected_checkpoint"]).resolve()
    if not checkpoint.is_relative_to(run_directory):
        raise ValueError("Checkpoint must belong to this run")
    if (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        != metadata["selected_checkpoint_sha256"]
    ):
        raise ValueError("Selected checkpoint hash does not match")
    for image_path in metadata["image_paths"]:
        path = (run_directory / image_path).resolve()
        if not path.is_relative_to(run_directory) or not path.is_file():
            raise ValueError(f"Reference image missing or outside run: {image_path}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    result = validate_run(parser.parse_args().run_directory)
    print(f"Verified reference tensor {result['shape']} and selected checkpoint hash")
