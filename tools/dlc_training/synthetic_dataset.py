"""Offline geometry-file adapter; original geometry and visibility remain authoritative."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dlc_pixel_coordinates(projected, projection_valid, image_size):
    """Convert edge-origin pixels to integer raster centres at the DLC boundary.

    DLC's Albumentations xy transforms flip x to (width - 1) - x and rotate
    around ((width - 1)/2, (height - 1)/2). Renderer centres are half-integers.
    Physical visibility is deliberately not an input to this conversion.
    """
    values = np.array(projected, dtype=np.float64, copy=True) - 0.5
    if values.shape[-1] != 2 or projection_valid.shape != values.shape[:-1]:
        raise ValueError(
            "Expected [image, feature, xy] coordinates and matching projection validity"
        )
    width, height = image_size
    labeled = (
        projection_valid
        & np.isfinite(values).all(axis=-1)
        & (values[..., 0] > 0)
        & (values[..., 0] < width)
        & (values[..., 1] > 0)
        & (values[..., 1] < height)
    )
    values[~labeled] = np.nan
    return values, labeled


def read_dataset(source: Path):
    source = source.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    membership = json.loads((source / "split_membership.json").read_text())
    digest = file_hash(source / "labels.npz")
    if digest != manifest["label_sha256"] or digest != membership["labels_sha256"]:
        raise ValueError("Geometry labels hash mismatch")
    if (
        manifest["pixel_convention"]
        != "origin at top-left image edge; pixel centres at half-integers"
    ):
        raise ValueError("Unsupported source pixel convention")
    with np.load(source / "labels.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    names = arrays["feature_names"].tolist()
    if names != manifest["feature_definition"]["feature_names"] or len(
        set(names)
    ) != len(names):
        raise ValueError(
            "Feature order must match the geometry manifest and contain unique identities"
        )
    count = len(arrays["image_names"])
    if (
        arrays["projected_xy_pixels"].shape != (count, len(names), 2)
        or arrays["projection_valid"].shape != (count, len(names))
        or arrays["visibility"].shape != (count, len(names))
        or arrays["projection_valid"].dtype != np.bool_
        or arrays["visibility"].dtype != np.bool_
    ):
        raise ValueError("Geometry tensor shape or mask dtype mismatch")
    if len({str(name) for name in arrays["image_names"]}) != count:
        raise ValueError("Image paths must be unique")
    split_names = set(membership["splits"])
    if split_names == {"training", "development", "final_evaluation"}:
        selected_names = ("training", "development", "final_evaluation")
    elif split_names == {"training", "evaluation", "excluded_guard_band"}:
        selected_names = ("training", "evaluation", "excluded")
    else:
        raise ValueError(
            "Unsupported split membership; require explicit disjoint groups"
        )
    groups = {
        name: np.asarray(
            membership["splits"]["excluded_guard_band" if name == "excluded" else name][
                "image_indices"
            ],
            dtype=np.int64,
        )
        for name in selected_names
    }
    combined = np.concatenate(list(groups.values()))
    if not np.array_equal(np.sort(combined), np.arange(count)):
        raise ValueError("Split membership must partition images exactly once")
    for name in arrays["image_names"]:
        image = (source / str(name)).resolve()
        if (
            not image.is_relative_to(source)
            or file_hash(image) != manifest["image_sha256"][str(name)]
        ):
            raise ValueError(f"Image path or hash mismatch: {name}")
        with Image.open(image) as pixels:
            if pixels.size != tuple(manifest["image_size"]) or pixels.mode != "RGB":
                raise ValueError(f"Image dimensions or color mode mismatch: {name}")
    coordinates, labeled = dlc_pixel_coordinates(
        arrays["projected_xy_pixels"],
        arrays["projection_valid"],
        manifest["image_size"],
    )
    return manifest, membership, arrays, groups, coordinates, labeled


def training_selection(groups):
    """DLC's training-time test split uses development, never reserved final data."""
    validation_name = "development" if "development" in groups else "evaluation"
    selected = np.sort(np.concatenate((groups["training"], groups[validation_name])))
    return validation_name, selected
